---
classes: wide2
title: "从零实现 LLM Inference：062. Triton KV append identity-pos（full batch 但 pos 不一致）"
excerpt: "上一版 full-batch KV-append 只覆盖 pos 常量的稳态；但只要 prompt 长度不一致，decode 的 pos 就会按 request 分叉，仍然要构造 batch_idx/pos 这类 index tensor。这版补一个 identity batch 的 Triton kernel：隐式 src_b=b，只需要 block_idx + pos_t，省掉 batch_idx 的分配。"
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

上一版（061）加了 `kv_append_triton_full_batch()`，把最常见的稳态（`fast_batch_idx=[0..B-1]` 且 `pos` 常量）做成：

- 不需要 `batch_idx`
- 不需要 `pos_t`

但只要 prompt 长度不一致，decode 之后每个 request 的 `pos` 会一直保持“错位”（差值固定），于是 `pos` 就不再是常量；虽然 `fast_batch_idx` 仍然是 identity，但我们会退回到通用 Triton kernel：

- 每 step 还要构造 `idx_t`（`batch_idx`）
- `pos_t` 也还是要构造

这版的目标就是：**在 identity batch + pos 不一致时，把 `idx_t` 从热路径挪出去**。

## 代码变更

### `roseinfer/kv_append_triton.py`

新增一个更窄的 kernel：`_kv_append_identity_pos_kernel()`，语义很简单：

- `src_b = b`（identity）
- `pos = pos_ptr[b]`（per-request）
- 写入 `k/v_cache[block_idx[b], h, pos, d]`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/kv_append_triton.py b/rosellm/roseinfer/kv_append_triton.py
@@
+@triton.jit
+def _kv_append_identity_pos_kernel(
+    k_cache_ptr, v_cache_ptr,
+    key_ptr, value_ptr,
+    block_idx_ptr, pos_ptr,
+    ...
+):
+    b = tl.program_id(0)
+    h = tl.program_id(1)
+    pid_d = tl.program_id(2)
+    ...
+    blk = tl.load(block_idx_ptr + b)
+    pos = tl.load(pos_ptr + b)
+    ...
+    tl.store(...)
+
+def kv_append_triton_identity_pos(..., block_idx: torch.Tensor, pos: torch.Tensor) -> None:
+    ...
+    grid = (B, H, triton.cdiv(D, block_d))
+    _kv_append_identity_pos_kernel[grid](..., block_idx, pos, ...)
```

### `roseinfer/engine.py`

在 `KVBlockManager.append_token_batch()` 里加一个分支：

- `full_fast` 恒成立（identity batch）
- `const_pos` 为 false 时，走 `kv_append_triton_identity_pos()`，省掉 `idx_t`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
 if use_triton_full_batch:
     ...  # 061 的 full-batch（pos 常量）
+elif use_triton_identity_pos:
+    blk_t = torch.tensor(fast_block_idx, device=device, dtype=torch.int32)
+    pos_t = torch.tensor(fast_pos, device=device, dtype=torch.int32)
+    kv_append_triton_identity_pos(
+        k_cache_layer=k_layer,
+        v_cache_layer=v_layer,
+        key_new=key_new,
+        value_new=value_new,
+        block_idx=blk_t,
+        pos=pos_t,
+    )
 elif use_triton:
     ...  # 通用 kernel（需要 idx_t + pos_t）
```

### `tests/test_triton_kv_append.py`

参数化覆盖 3 条路径：

- `full_batch`：pos 常量
- `identity_pos`：pos 不一致
- `generic`：禁用 full-batch 后走通用 kernel（回归兜底）

## 运行

```bash
pytest -q
```

```text
....................................                                     [100%]
36 passed, 1 warning in 2.87s
```

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 强制制造 “pos 不一致”：

- `--prompt-repeats "1,4"`：让一半请求的 prompt 更长（prefill 后 pos 错位）
- `--no-prefix-cache`：专注看 decode steady-state 的 KV append 热路径

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --prompt-repeats '1,4' \
  --pretok --tokenize-workers 0 \
  --num-requests 128 --max-new-tokens 128 \
  --submit-interval-ms 0 \
  --max-batch-size 128 --prefill-max-batch-size 128 \
  --prefill-admission-policy fifo \
  --paged-attn --no-prefix-cache --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 3
```

Before（3 次 run）：

```text
run1 Throughput: 14517.43 tokens/s | TPOT p50: 8.38 ms/token
run2 Throughput: 14724.32 tokens/s | TPOT p50: 8.27 ms/token
run3 Throughput: 14471.46 tokens/s | TPOT p50: 8.41 ms/token
avg  Throughput: 14571.07 tokens/s
```

After（3 次 run）：

```text
run1 Throughput: 14924.67 tokens/s | TPOT p50: 8.14 ms/token
run2 Throughput: 14588.83 tokens/s | TPOT p50: 8.35 ms/token
run3 Throughput: 14863.42 tokens/s | TPOT p50: 8.18 ms/token
avg  Throughput: 14792.31 tokens/s
```

平均吞吐 **+1.52%**（`14571.07 -> 14792.31 tokens/s`）。

## 结论

- 061 覆盖 “pos 常量” 的稳态，这版覆盖 “pos 不一致但 batch 仍是 identity” 的稳态：两者组合后，decode 热路径里 `batch_idx` 基本可以不再分配。
- 这类优化的本质都是一样的：**减少每 step 的小 tensor 构造和索引开销**，把它们从热路径里挪走。
- 下一步如果继续抠：`blk_t/pos_t` 这两个 index tensor 仍然是每 step 构造的，可以考虑做成可复用 buffer（尤其是大 batch + 长 decode 场景）。

