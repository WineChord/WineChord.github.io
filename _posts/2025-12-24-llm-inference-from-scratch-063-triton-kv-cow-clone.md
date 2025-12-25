---
classes: wide2
title: "从零实现 LLM Inference：063. Triton batched KV COW clone（替换 index_select/index_copy）"
excerpt: "prefix cache hit 后，多条 session 会共享同一批 KV blocks；第一次 decode 写入时如果 last block 还没写满，就会触发 COW：先 clone block 再 append token。原实现用 index_select/index_copy 搬整块 KV；这版加一个 Triton kernel 做 batched block copy，把这段从通用 gather/scatter 换成纯 memcpy 风格。"
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

prefix cache 的收益来自 “共享 prefix 的 KV blocks”。但共享不是免费的：当多个 request 共享同一个 **未写满的 last block** 时，第一次 decode 写入会触发 **copy-on-write（COW）**：

1) 给每条 session 分配一个新 block（保持 prefix cache entry 的 block 不被污染）
2) 把旧 block 的内容 copy 到新 block
3) 在新 block 的 `pos=old_len` 写入新 token 的 KV

之前 `KVBlockManager.append_token_batch()` 的 clone 用的是：

- `index_select(0, old_blk_t)` + `index_copy_(0, new_blk_t, ...)`

这属于很通用的 gather/scatter，功能强但也会带来额外开销。这里我们只需要做一件事：**把 `[H, BS, D]` 这一整块 KV 从 src block 拷到 dst block**。

所以这版直接做成 Triton “memcpy 风格”的 kernel。

## 代码变更

### `roseinfer/kv_clone_triton.py`

新增一个 kernel：grid 组织成 `(N, H, ceil_div(BS*D, BLOCK))`，每个 program 拷贝一个 `(H, tile)` 的子块，同时把 K/V 一起搬。

核心 diff：

```diff
diff --git a/rosellm/roseinfer/kv_clone_triton.py b/rosellm/roseinfer/kv_clone_triton.py
new file mode 100644
@@
+@triton.jit
+def _kv_clone_blocks_kernel(k_cache_ptr, v_cache_ptr, src_block_idx_ptr, dst_block_idx_ptr, ...):
+    n = tl.program_id(0)
+    h = tl.program_id(1)
+    pid = tl.program_id(2)
+    td = pid * BLOCK + tl.arange(0, BLOCK)
+    t = td // D
+    d = td - t * D
+    src_blk = tl.load(src_block_idx_ptr + n)
+    dst_blk = tl.load(dst_block_idx_ptr + n)
+    k = tl.load(k_cache_ptr + src_off, mask=..., other=0)
+    v = tl.load(v_cache_ptr + src_off, mask=..., other=0)
+    tl.store(k_cache_ptr + dst_off, k, mask=...)
+    tl.store(v_cache_ptr + dst_off, v, mask=...)
+def kv_clone_blocks_triton(k_cache_layer, v_cache_layer, src_block_idx, dst_block_idx) -> None:
+    _kv_clone_blocks_kernel[grid](...)
```

### `roseinfer/engine.py`

在 COW clone 分支里，如果 CUDA + Triton 可用，就用 `kv_clone_blocks_triton()`；否则回退到原来的 `index_select/index_copy_`。

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
 if cow_old_block_idx:
-    old_blk_t = torch.tensor(cow_old_block_idx, device=device, dtype=torch.long)
-    new_blk_t = torch.tensor(cow_new_block_idx, device=device, dtype=torch.long)
-    k_src = k_layer.index_select(0, old_blk_t)
-    v_src = v_layer.index_select(0, old_blk_t)
-    k_layer.index_copy_(0, new_blk_t, k_src)
-    v_layer.index_copy_(0, new_blk_t, v_src)
+    if use_triton_clone:
+        old_blk_t = torch.tensor(cow_old_block_idx, device=device, dtype=torch.int32)
+        new_blk_t = torch.tensor(cow_new_block_idx, device=device, dtype=torch.int32)
+        kv_clone_blocks_triton(k_cache_layer=k_layer, v_cache_layer=v_layer, src_block_idx=old_blk_t, dst_block_idx=new_blk_t)
+    else:
+        ...  # 原实现
```

### `tests/test_triton_kv_cow_clone.py`

补一个 CUDA 单测：构造 refcount>1 的共享 last block，调用 `append_token_batch()`，检查：

- base block 内容不变
- 新 block 前缀被正确拷贝
- 新 token 被正确写入

## 运行

```bash
pytest -q
```

```text
.....................................                                    [100%]
37 passed, 1 warning in 2.70s
```

## Benchmark（HF GPT-2 / streaming / prefix cache hit）

这组 benchmark 的重点是制造 “prefix cache hit + COW clone”：

- warmup 跑一次把 cache 填满
- measured run 里大多数请求走 cache attach，然后第一 token 触发 COW clone
- `--prefill-max-batch-size 1`：让 cache hit 更稳定（逐个 admission）
- `--max-new-tokens 1`：只看 “prefill->first token / TTFT” 这一步

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --pretok --tokenize-workers 0 \
  --num-requests 256 --max-new-tokens 1 \
  --submit-interval-ms 0 \
  --max-batch-size 128 --prefill-max-batch-size 1 \
  --prefill-admission-policy fifo \
  --paged-attn --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 3
```

Before（3 次 run）：

```text
Throughput: 3368.23 / 3391.88 / 3423.10 tokens/s (avg 3394.40)
Prefill->first token p99: 0.59 / 0.55 / 0.51 ms (avg 0.55)
TTFT p99: 26.44 / 28.22 / 27.97 ms (avg 27.54)
```

After（3 次 run）：

```text
Throughput: 3538.89 / 3461.56 / 3379.08 tokens/s (avg 3459.84)
Prefill->first token p99: 0.52 / 0.49 / 0.47 ms (avg 0.49)
TTFT p99: 25.14 / 28.82 / 28.93 ms (avg 27.63)
```

结论更看重两点：

- `Prefill->first token p99` 从 **0.55ms -> 0.49ms（~ -10%）**：更接近我们要优化的那段 “clone + append” 热路径
- completion throughput 平均 **+1.93%**（`3394.40 -> 3459.84 tokens/s`）

TTFT 仍然主要被 admission/queue wait 主导（prefill-max-batch-size=1 的代价），所以 TTFT 的均值/尾部不是这版最敏感的指标。

## 结论

- COW clone 本质上就是 KV block 的拷贝，适合用 Triton 写成更“直白”的 block copy kernel。
- 这个优化更偏向 prefix cache hit 的场景：让 “hit 后第一 token” 更干净（prefill->first token 的尾巴更小）。
- 下一步如果继续抠：可以考虑把 `old_blk_t/new_blk_t` 也做成可复用 buffer，进一步减少每 step 的小 tensor 构造开销。

