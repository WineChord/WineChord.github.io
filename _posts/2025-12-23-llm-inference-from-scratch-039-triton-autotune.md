---
classes: wide2
title: "从零实现 LLM Inference：039. Triton Autotune（paged-attn decode kernel）"
excerpt: "把 paged-attn decode kernel 的 num_warps/num_stages 从“拍脑袋常量”变成 Triton autotune；结果在 decode 热路径里反而回退了。"
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

paged attention 这条路径里，我们的 decode kernel 会被反复调用（每层一次、每个 token 一次）。

之前为了省事，我直接把 launch 参数写死成了 `num_warps=4`。

这一版想做一件“看起来很合理”的事情：给 kernel 加 `triton.autotune`，让它自己在启动时选一个最合适的 `num_warps/num_stages`，避免我们硬编码。

## 代码变更

### `rosetrainer/paged_attention.py`

核心思路：

- 给 `_paged_attn_decode_kernel` 加 `@triton.autotune(...)`
- 提供一组 configs（只调 `num_warps/num_stages`）
- 用 `key=["H","D","BLOCK_SIZE","MAX_BLOCKS"]` 做 shape cache

```diff
diff --git a/rosellm/rosetrainer/paged_attention.py b/rosellm/rosetrainer/paged_attention.py
@@
+    _PAGED_ATTN_AUTOTUNE_CONFIGS = [
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ]
+
+    @triton.autotune(
+        configs=_PAGED_ATTN_AUTOTUNE_CONFIGS,
+        key=["H", "D", "BLOCK_SIZE", "MAX_BLOCKS"],
+    )
     @triton.jit
     def _paged_attn_decode_kernel(...):
@@
-        num_warps=4,
 ```

## 运行

```bash
pytest -q
```

```text
......................                                                   [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
22 passed, 1 warning in 1.65s
```

## Benchmark（HF GPT-2）

命令保持不变（收敛到 paged decode 热路径）：

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn
```

### Before（固定 num_warps=4）

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 3.305404/3.303470 s
Total time p50/mean: 3.478114/3.478629 s
Throughput(completion,decode) p50/mean: 9913.46/9919.28 tokens/s
Throughput(completion,total) p50/mean: 9421.20/9419.81 tokens/s
TTFT p50/mean: 2.73/2.73 ms
TPOT p50/mean: 6.50/6.50 ms/token
Latency p50/mean: 3324.15/3326.38 ms
```

### After（Triton autotune）

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 3.386243/3.388143 s
Total time p50/mean: 3.558176/3.561974 s
Throughput(completion,decode) p50/mean: 9676.80/9671.41 tokens/s
Throughput(completion,total) p50/mean: 9209.21/9199.43 tokens/s
TTFT p50/mean: 2.69/2.70 ms
TPOT p50/mean: 6.66/6.67 ms/token
Latency p50/mean: 3408.16/3409.43 ms
```

## 结论

这次“自动选 launch 参数”在这个 workload 上是 **负收益**：

- decode 吞吐（mean）：`9919.28 → 9671.41 tokens/s`（约 **-2.5%**）
- TPOT（mean）：`6.50 → 6.67 ms/token`（约 **+2.6%**）

直觉上原因也不难理解：paged decode 的 kernel 调用次数非常高，`autotune` 的 wrapper/dispatch 本身也有固定开销；当单次 kernel 足够小、launch 足够密的时候，这点开销就会被放大成可见的回退。
