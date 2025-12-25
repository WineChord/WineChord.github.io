---
classes: wide2
title: "从零实现 LLM Inference：051. Fast TopK/TopP Sampling（避开 full-vocab sort/scatter）"
excerpt: "之前 top_k+top_p 采样每步都会对整个 vocab 做 sort/gather，开销巨大。把采样改成直接在 sorted topk 空间里做 top-p + multinomial，并顺手把 top_k clamp 到 vocab size。"
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

前面几版我们一直在抠 streaming / scheduler 的 CPU overhead，但 decode 里还有一块非常容易被忽略：**采样**。

如果你打开 `--do-sample --top-k 40 --top-p 0.9`，我们之前的实现每一步都会：

1) 先做一次 `top_k`（但还是保留成 `[B, V]` 的 full logits）
2) `top_p` 里再对 `[B, V]` 做一次全量 sort（V=50257）
3) 为了回到原 vocab 顺序，还要再 sort / gather 一次

这在 batch 大、decode token 多的时候非常伤：算子本身不是模型 forward，但 **V 维度太大**，sort/gather 的固定成本会被放大成吞吐瓶颈。

这一版的目标很明确：**top_k>0 的情况下，top_p 采样不再走 full-vocab sort + scatter 回 vocab**，直接在 sorted topk 空间里完成筛选 + multinomial。

## 代码变更

### `roseinfer/engine.py`

1) `_top_k_logits` 顺手补了一个 guard：`top_k >= vocab` 时直接返回 logits（避免 top_k 设太大直接 crash）。

2) `_sample_next_token_batch` 改成分三条路径：

- `top_p` 关闭（<=0 或 >=1）
  - `top_k` 关闭：直接对 `[B, V]` softmax + multinomial
  - `top_k` 打开：只对 `[B, K]`（K=top_k）softmax + multinomial，然后用 `topk_idx` 映射回 token id

- `top_p` 打开（0<top_p<1）
  - 先做一次 `torch.topk` 得到 `sorted_logits/sorted_idx`（K=top_k 或者 K=V）
  - 在 `[B, K]` 上做 softmax/cumsum，mask 掉 top_p 以外的部分
  - 直接在 `[B, K]` 上 multinomial，然后 gather 回 token id

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
 def _top_k_logits(self, logits, top_k):
-    if top_k <= 0:
+    vocab = int(logits.size(-1))
+    top_k = int(top_k)
+    if top_k <= 0 or top_k >= vocab:
         return logits
@@
 def _sample_next_token_batch(self, logits, temperature, top_k, top_p, do_sample):
     if not do_sample or temperature <= 0.0:
         return torch.argmax(logits, dim=-1)
     scaled = logits / float(temperature)
+    vocab = int(scaled.size(-1))
+    top_k = int(top_k)
+
+    # top_p disabled
+    if top_p <= 0.0 or top_p >= 1.0:
+        if top_k <= 0 or top_k >= vocab:
+            probs = torch.softmax(scaled, dim=-1).clamp_min(1e-9)
+            return torch.multinomial(probs, 1).squeeze(-1)
+        k = min(top_k, vocab)
+        topk_logits, topk_idx = torch.topk(scaled, k, dim=-1)
+        probs = torch.softmax(topk_logits, dim=-1).clamp_min(1e-9)
+        choice = torch.multinomial(probs, 1).squeeze(-1)
+        return topk_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
+
+    # top_p enabled: sample in sorted topk space (avoid full-vocab scatter)
+    k = vocab if top_k <= 0 else min(top_k, vocab)
+    sorted_logits, sorted_idx = torch.topk(scaled, k, dim=-1)
+    probs = torch.softmax(sorted_logits, dim=-1)
+    cum = torch.cumsum(probs, dim=-1)
+    mask = cum > float(top_p)
+    mask[..., 0] = False
+    probs = probs.masked_fill(mask, 0.0).clamp_min(1e-9)
+    choice = torch.multinomial(probs, 1).squeeze(-1)
+    return sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
```

### `tests/test_fast_topk_sampling.py`

补两个最小测试：

- `top_k` 大于 vocab 不 crash（并且 `_top_k_logits` 直接返回原 logits）
- `_sample_next_token_batch(..., top_k=10000, do_sample=True)` 不 crash，输出 id 在 vocab 范围内

## 运行

```bash
pytest -q
```

```text
..............................                                           [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 1 warning in 2.16s
```

## Benchmark（HF GPT-2 / streaming / sampling）

为了把采样开销放大，我用：

- `num_requests=64`
- `max_new_tokens=128`
- `max_batch_size=64`（64 个请求一起 decode）
- 打开采样：`--do-sample --top-k 40 --top-p 0.9`

（我这里加了 offline 兜底，避免偶发网络抖动：）

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Before（旧实现）

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --num-requests 64 --max-new-tokens 128 \
  --max-batch-size 64 --prefill-max-batch-size 64 \
  --tokenize-workers 4 \
  --no-stop-on-eos --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1 \
  --do-sample --top-k 40 --top-p 0.9
```

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 4
Stream interval: 1
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 0.007751 s
add_request latency p50/p95/p99: 0.01/0.01/0.04 ms
Tokenize p50/p95/p99: 0.11/0.25/0.42 ms
Queue wait (post-tok) p50/p95/p99: 16.21/19.48/19.92 ms
Prefill->first token p50/p95/p99: 37.75/37.82/37.83 ms
TTFT p50/p95/p99: 54.07/57.43/57.91 ms
TPOT p50/p95/p99: 30.25/30.25/30.30 ms/token
ITL p50/p95/p99: 34.79/37.26/40.32 ms
Latency p50/p95/p99: 3899.87/3901.07/3902.24 ms
Throughput (completion,total): 2097.43 tokens/s
```

### After（fast topk/top-p）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 4
Stream interval: 1
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 0.006588 s
add_request latency p50/p95/p99: 0.01/0.01/0.03 ms
Tokenize p50/p95/p99: 0.07/0.23/0.32 ms
Queue wait (post-tok) p50/p95/p99: 9.89/12.86/13.12 ms
Prefill->first token p50/p95/p99: 38.68/38.77/38.78 ms
TTFT p50/p95/p99: 48.66/51.57/52.04 ms
TPOT p50/p95/p99: 26.01/26.01/26.04 ms/token
ITL p50/p95/p99: 29.83/33.01/34.33 ms
Latency p50/p95/p99: 3354.78/3357.08/3358.18 ms
Throughput (completion,total): 2438.21 tokens/s
```

## 结论

这版本质是在“采样层”把全量 vocab 的 sort/scatter 去掉了，直接在 `[B, K]`（K=top_k）里做 top-p + multinomial：

- `Throughput`: 2097.43 → 2438.21 tokens/s（+16.3%）
- `TPOT`: 30.25 → 26.01 ms/token（-14.0%）
- `Latency p99`: 3902.24 → 3358.18 ms（-13.9%）

这类优化非常典型：**模型 forward 没变，但把“后处理固定成本”压下去，吞吐就会明显上来**。后面如果要继续往上走，可以把 sampler 做成 Triton/CUDA kernel（或复用 cutlass/cub 的思路），再进一步降低这块的 GPU/launch 开销。

