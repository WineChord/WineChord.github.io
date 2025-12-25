---
classes: wide2
title: "从零实现 LLM Inference：061. Triton KV append full-batch fast path（少分配 index tensor）"
excerpt: "batch decode 稳态里，经常满足 fast_batch_idx 是 [0..B-1] 且同一步 pos 对整个 batch 是常量；这时 Triton KV append 还在每 step 构造 batch_idx/pos 这类小 tensor。这个 PR 新增 full-batch Triton kernel：隐式 identity batch + 标量 pos，只保留 block_idx，减少热路径分配，并保持通用路径不变。"
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

`KVBlockManager.append_token_batch()` 在 decode 阶段会被每层、每 step 调很多次。即便已经走 Triton fast path，原来仍有一笔“细碎但频繁”的开销：

- 每 step 都会构造 `idx_t/pos_t`（把 Python list 转成 CUDA int32 tensor）
- 而在一些很常见的稳态里，这些 index tensor 其实是冗余的：
  - `fast_batch_idx` 是 identity：`[0..B-1]`
  - 同一步里 `pos` 对整个 batch 是常量（所有 request 的长度对齐）

这版做的事情很小：在满足上述条件时，走一个更专用的 KV 写入 kernel。

## 代码变更

### `roseinfer/kv_append_triton.py`

新增 full-batch kernel + wrapper：

- **不需要** `batch_idx`（隐式 `src_b = b`）
- **不需要** `pos_t`（传标量 `pos`）
- 只保留 `block_idx[b]`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/kv_append_triton.py b/rosellm/roseinfer/kv_append_triton.py
@@
+TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH = int(
+    os.environ.get(
+        "ROSELLM_TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH", str(TRITON_KV_APPEND_MIN_BATCH)
+    )
+)
+
+@triton.jit
+def _kv_append_full_batch_kernel(
+    k_cache_ptr, v_cache_ptr,
+    key_ptr, value_ptr,
+    block_idx_ptr, pos,
+    ...
+):
+    b = tl.program_id(0)
+    h = tl.program_id(1)
+    pid_d = tl.program_id(2)
+    ...
+    blk = tl.load(block_idx_ptr + b).to(tl.int32)
+    p = tl.full((), pos, tl.int32)
+    ...
+    tl.store(...)
+
+def kv_append_triton_full_batch(..., block_idx: torch.Tensor, pos: int) -> None:
+    ...
+    grid = (B, H, triton.cdiv(D, block_d))
+    _kv_append_full_batch_kernel[grid](..., block_idx, pos=int(pos), ...)
```

这里把 `ROSELLM_TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH` 默认设成和 `ROSELLM_TRITON_KV_APPEND_MIN_BATCH` 一致：默认只在“本来就会用 Triton 的大 batch”里再额外尝试 full-batch fast path。

### `roseinfer/engine.py`

在 `append_token_batch()` 里加一个 very small 的 gate：

- `full_fast`：fast batch 覆盖全量且是 identity
- `const_pos`：同一步 `pos` 对 batch 常量

满足条件时直接调用 `kv_append_triton_full_batch()`。

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
     full_fast = ...
     pos0 = fast_pos[0]
     const_pos = all(p == pos0 for p in fast_pos)
@@
     use_triton_full_batch = (
         TRITON_AVAILABLE
         and USE_TRITON_KV_APPEND
         and full_fast
         and const_pos
         and len(fast_batch_idx) >= TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH
     )
@@
     if use_triton_full_batch:
         blk_t = torch.tensor(fast_block_idx, device=device, dtype=torch.int32)
         kv_append_triton_full_batch(
             k_cache_layer=k_layer,
             v_cache_layer=v_layer,
             key_new=key_new,
             value_new=value_new,
             block_idx=blk_t,
             pos=pos0,
         )
     elif use_triton:
         ...  # 原来的通用 Triton 路径不变
```

### `tests/test_triton_kv_append.py`

把单测参数化：同时覆盖 full-batch 和通用 Triton kernel 的 correctness。

## 运行

```bash
pytest -q
```

```text
...................................                                      [100%]
35 passed, 1 warning in 2.83s
```

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 关掉 prefix cache，专注看 decode steady-state（KV append 会被调用很多次）。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --pretok --tokenize-workers 0 \
  --num-requests 128 --max-new-tokens 128 \
  --submit-interval-ms 0 \
  --max-batch-size 128 --prefill-max-batch-size 128 \
  --prefill-admission-policy fifo \
  --paged-attn --no-prefix-cache --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 3
```

Before（3 次 run）：

```text
run1 Throughput: 14351.31 tokens/s | TPOT p50: 8.48 ms/token
run2 Throughput: 14110.61 tokens/s | TPOT p50: 8.66 ms/token
run3 Throughput: 14798.32 tokens/s | TPOT p50: 8.22 ms/token
avg  Throughput: 14420.08 tokens/s
```

After（3 次 run）：

```text
run1 Throughput: 15075.12 tokens/s | TPOT p50: 8.06 ms/token
run2 Throughput: 14403.43 tokens/s | TPOT p50: 8.46 ms/token
run3 Throughput: 14466.05 tokens/s | TPOT p50: 8.42 ms/token
avg  Throughput: 14648.20 tokens/s
```

平均吞吐 **+1.58%**（`14420.08 -> 14648.20 tokens/s`）。这类优化的特点就是：每 step 省一点点（少分配/少拷贝），在长 decode 和多层叠加后会反映到 TPOT/吞吐。

## 结论

- Triton kernel 本身再快，如果热路径里还有“每 step 构造小 tensor”这种 Python 开销，也会吃掉不少收益。
- full-batch fast path 的关键点不是“换一个更神秘的 kernel”，而是 **把可省的 index tensor 从热路径挪出去**。
- 下一步如果继续抠 KV 写入：可以把 “full_fast 但 pos 不一致” 也做成专用 kernel（只去掉 `idx_t`），覆盖更普遍的 online 负载形态。

