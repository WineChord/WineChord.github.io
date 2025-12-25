---
classes: wide2
title: "从零实现 LLM Inference：058. Prefix Cache logits 常驻 device（减少 hit 的 CPU/GPU 拷贝）"
excerpt: "prefix cache hit 时我们还在把 last_logits 从 CPU 拷回 GPU；这版把 entry 的 last_logits 直接存成 device 上的一份小 clone，hit 变成真正的零拷贝。"
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

prefix cache 的语义里，除了 KV blocks 以外，还有一件很关键的东西：**prefill 的 last_logits**。

因为无论 exact hit 还是 longest-prefix reuse，最终都要从 last_logits 里采样出 “prefill 之后的第一个 token”。

之前的实现把 `last_logits` 固定存 CPU：

- `put()`：GPU -> CPU
- `attach()`：CPU -> GPU

KV 已经都在 GPU 了，但 logits 这一小段还在来回搬。对小模型来说，这类“细碎拷贝”会直接反映到 TTFT。

这版就做一件事：**PrefixCacheEntry 的 last_logits 直接存成 device 上的一份小 clone**（避免 slice 引用把整块 logits 留住），hit 变成真正的零拷贝。

## 代码变更

### `roseinfer/engine.py`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
 class PrefixCacheEntry:
@@
-        self.last_logits = last_logits.detach().to("cpu")
+        self.last_logits = last_logits.detach().clone()
```

这里必须用 `clone()`：`last_logits` 经常是从 batch logits 里 slice 出来的一行，如果不 clone，就会把整个 `[B, vocab]` 的 storage 一起挂在 cache 里。

## 运行

```bash
pytest -q
```

```text
.................................                                        [100%]
33 passed, 1 warning in 2.70s
```

## Benchmark（HF GPT-2 / streaming）

为了让 prefix cache 的 hit 路径充分出现，这里强制 `--prefill-max-batch-size 1`（避免同一批 admission 内部的 dup-of 直接共享绕过 cache），并且 `--max-new-tokens 1`（只看 TTFT，不跑 decode）。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --pretok --tokenize-workers 0 \
  --num-requests 1024 --max-new-tokens 1 \
  --submit-interval-ms 6 --submit-schedule absolute \
  --max-batch-size 1 --prefill-max-batch-size 1 \
  --prefill-admission-policy fifo \
  --paged-attn --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 1
```

Before：

```text
add_request latency p50/p95/p99: 0.06/0.10/0.15 ms
Prefill->first token p50/p95/p99: 0.26/0.46/0.63 ms
TTFT p50/p95/p99: 0.38/0.73/0.84 ms
```

After：

```text
add_request latency p50/p95/p99: 0.03/0.06/0.07 ms
Prefill->first token p50/p95/p99: 0.24/0.37/0.48 ms
TTFT p50/p95/p99: 0.32/0.47/0.62 ms
```

## 结论

- prefix cache 不是只有 KV，**last_logits 也要当成一等公民**。
- 把 logits 留在 device 上，hit 路径就不再付 CPU/GPU 拷贝税，TTFT 的尾巴也更干净。
- 后面如果继续在 hit 路径上抠：可以进一步看 “attach 时的 incref 账单”（block 数越多越明显），以及 longest-prefix reuse 的 suffix replay 怎么更高效。

