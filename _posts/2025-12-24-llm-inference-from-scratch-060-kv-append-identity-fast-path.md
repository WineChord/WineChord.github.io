---
classes: wide2
title: "从零实现 LLM Inference：060. KV append identity fast path（少做一次 index_select）"
excerpt: "append_token_batch 里 fast_batch_idx 很多时候就是 [0..B-1]；之前每层都会 index_select 把 key_new/value_new 重新拷一遍，还会构造 pos_t。这个小改动在 identity batch 时直接复用 key_new/value_new，并且 pos 常量时不再分配 pos_t，让 decode 的 KV 写入更轻。"
categories:
  - LLM
  - Inference
tags:
  - LLM
  - Inference
toc: true
toc_sticky: true
mathjax: true
---

`KVBlockManager.append_token_batch()` 的 baseline 写法比较直接：

- 先把 fast batch 的 `batch_idx` 做成 tensor
- `key_new/value_new.index_select(0, batch_idx)` 得到 `k_src/v_src`
- 再把 `k_src/v_src` scatter 写进 KV cache

但在 decode 的大多数 step 里，`fast_batch_idx` 其实就是一个很常见的形态：

- **identity**：`[0, 1, 2, ..., B-1]`（batch 没有被拆成 fast/slow 两块）
- **const pos**：同一个 step 下，`pos` 对整个 batch 是常量

这时 `index_select` 就纯属多做一次拷贝；`pos_t` 也是不必要的分配。

这版 PR 就做两件小事：

1) `fast_batch_idx` 是 identity 时：直接用 `key_new/value_new`，跳过 `index_select`
2) `fast_pos` 是常量时：用标量 `pos0` 直接索引，跳过 `pos_t` 的构造

## 代码变更

### `roseinfer/engine.py`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
-                idx_t = torch.tensor(fast_batch_idx, device=device, dtype=torch.long)
                 blk_t = torch.tensor(fast_block_idx, device=device, dtype=torch.long)
-                pos_t = torch.tensor(fast_pos, device=device, dtype=torch.long)
-                k_src = key_new.index_select(0, idx_t)
-                v_src = value_new.index_select(0, idx_t)
-                k_layer[blk_t, :, pos_t, :] = k_src
-                v_layer[blk_t, :, pos_t, :] = v_src
+                full_fast = ...  # fast_batch_idx == [0..B-1] && no slow
+                if full_fast:
+                    k_src = key_new
+                    v_src = value_new
+                else:
+                    idx_t = torch.tensor(fast_batch_idx, device=device, dtype=torch.long)
+                    k_src = key_new.index_select(0, idx_t)
+                    v_src = value_new.index_select(0, idx_t)
+
+                pos0 = fast_pos[0]
+                const_pos = all(p == pos0 for p in fast_pos)
+                if const_pos:
+                    k_layer[blk_t, :, pos0, :] = k_src
+                    v_layer[blk_t, :, pos0, :] = v_src
+                else:
+                    pos_t = torch.tensor(fast_pos, device=device, dtype=torch.long)
+                    k_layer[blk_t, :, pos_t, :] = k_src
+                    v_layer[blk_t, :, pos_t, :] = v_src
```

## 运行

```bash
pytest -q
```

```text
..................................                                       [100%]
34 passed, 1 warning in 2.72s
```

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 关掉 prefix cache，专注看 decode 的 steady-state（KV append 会被调用很多次）。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --pretok --tokenize-workers 0 \
  --num-requests 256 --max-new-tokens 256 \
  --submit-interval-ms 0 \
  --max-batch-size 16 --prefill-max-batch-size 16 \
  --prefill-admission-policy fifo \
  --paged-attn --no-prefix-cache --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 1
```

Before：

```text
TPOT p50/p95/p99: 66.77/66.86/66.89 ms/token
ITL p50/p95/p99: 66.01/71.79/82.06 ms
Throughput (completion,total): 3807.13 tokens/s
```

After：

```text
TPOT p50/p95/p99: 65.41/65.49/65.52 ms/token
ITL p50/p95/p99: 64.68/70.10/74.99 ms
Throughput (completion,total): 3886.10 tokens/s
```

TPOT p50 大约 **-2.0%**，吞吐大约 **+2.1%**，ITL p99 也更稳。

## 结论

- `index_select` 在 identity batch 上属于“重复拷贝”，能省就省。
- `pos` 常量时避免构造 `pos_t`，属于很典型的“把 Python 小对象和小 tensor 从热路径里挪出去”。
- 下一步如果继续抠 KV 写入：可以把 `blk_t/pos_t` 这类 index tensor 也做成可复用 buffer（尤其是 decode 稳态时它们变化很慢）。
