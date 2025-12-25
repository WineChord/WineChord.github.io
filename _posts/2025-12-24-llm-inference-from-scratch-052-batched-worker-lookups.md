---
classes: wide2
title: "从零实现 LLM Inference：052. Batched Worker Lookups（减少 streaming decode 的锁开销）"
excerpt: "SchedulerManager 的 worker loop 里每个 token 都会进一次 lock 取 q/detok/state；改成每步 decode 批量抓取一次，减少 lock acquire/release 和热点 dict 访问。"
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

上一版我们已经把 streaming 的一些显性 CPU overhead（比如 flush 频率、inflight 限流等）抠掉了，但 worker loop 里还有一个很“隐蔽”的固定成本：

- `OnlineScheduler.step()` 每一步会返回 `step_tokens: Dict[rid, token_id]`
- 我们要把 token 推到对应 request 的 `queue/detokenizer/stream_state`

之前的写法是 **每个 token 都 `with self._lock` 一次**，在 batch 大、decode token 多的时候会产生大量 lock acquire/release（并且会放大和 `add_request()` 的竞争）。

这一版做的事很单纯：**同一轮 decode step 里，把所有 rid 的引用一次性在 lock 下抓出来**，然后在 lock 外完成 detok/flush/queue put。

## 代码变更

### `roseinfer/server.py`

把 `run_decode_once()` 里两段热点循环：

- `for rid, token_id in step_tokens.items(): ...`
- `for rid in finished_ids: ...`

从“每个 rid 进一次 lock”，改成：

1) 先在 lock 下构造 `step_records/finished_records`（只做 dict get，不做任何重活）
2) 在 lock 外跑 detok + flush + `q.put()`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
 def run_decode_once() -> None:
     step_tokens = self.scheduler.step()
     finished_ids = self.scheduler.pop_finished_ids()
-
-    for rid, token_id in step_tokens.items():
-        with self._lock:
-            q = self._queues.get(rid)
-            detok = self._detoks.get(rid)
-            state = self._stream_states.get(rid)
-            token_ts = self._token_timestamps.get(rid) if ... else None
-        ...
-
-    for rid in finished_ids:
-        with self._lock:
-            q = self._queues.get(rid)
-            detok = self._detoks.get(rid)
-            state = self._stream_states.get(rid)
-        self.scheduler.discard_request(rid)
-        ...
+
+    step_records = []
+    finished_records = []
+    with self._lock:
+        for rid, token_id in step_tokens.items():
+            step_records.append((rid, int(token_id), self._queues.get(rid), ...))
+        for rid in finished_ids:
+            finished_records.append((int(rid), self._queues.get(rid), ...))
+
+    for rid, token_id, q, detok, state, token_ts in step_records:
+        ...
+
+    for rid, q, detok, state in finished_records:
+        self.scheduler.discard_request(rid)
+        ...
```

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

## Benchmark（HF GPT-2 / streaming）

为了让 `add_request()` 和 decode 过程有更明显的重叠，我这里用了：

- `--submit-interval-ms 20 --submit-schedule absolute`（64 个请求分 ~1.26s 提交完）
- `--pretok`（把 tokenization 从 scheduler manager 路径上拿掉，减少噪声）
- `--stream-interval 8`（降低 flush 频率，避免把结果完全变成 queue/IO benchmark）

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --pretok --tokenize-workers 0 \
  --stream-interval 8 \
  --num-requests 64 \
  --warmup-runs 1 --repeat-runs 1 \
  --submit-interval-ms 20 --submit-schedule absolute \
  --max-batch-size 64 --prefill-max-batch-size 64 \
  --max-new-tokens 128 \
  --no-stop-on-eos --no-prefix-cache
```

### Before

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Tokenize workers: 0
Stream interval: 8
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 1.260277 s
Submit interval/schedule: 20.000 ms / absolute
Submit lag p50/p95/p99: 0.06/0.11/0.12 ms
add_request latency p50/p95/p99: 0.03/0.05/0.06 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 5.72/16.27/20.86 ms
Prefill->first token p50/p95/p99: 3.33/3.99/4.26 ms
TTFT p50/p95/p99: 9.21/19.87/24.99 ms
TPOT p50/p95/p99: 20.99/22.00/22.02 ms/token
ITL p50/p95/p99: 22.92/24.49/26.05 ms
Latency p50/p95/p99: 2680.94/2805.04/2809.93 ms
Throughput (completion,total): 2156.57 tokens/s
```

### After（batched lookups）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Tokenize workers: 0
Stream interval: 8
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 1.260339 s
Submit interval/schedule: 20.000 ms / absolute
Submit lag p50/p95/p99: 0.07/0.16/0.17 ms
add_request latency p50/p95/p99: 0.03/0.06/0.08 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 4.29/15.24/20.70 ms
Prefill->first token p50/p95/p99: 3.37/4.08/5.53 ms
TTFT p50/p95/p99: 7.60/18.62/24.86 ms
TPOT p50/p95/p99: 21.09/22.10/22.12 ms/token
ITL p50/p95/p99: 22.55/24.62/25.21 ms
Latency p50/p95/p99: 2687.46/2815.67/2821.78 ms
Throughput (completion,total): 2150.25 tokens/s
```

## 结论

这版优化点非常“朴素”：把 decode 的 per-token lock 变成 per-step lock，本质是降低 worker loop 的 Python overhead。

在这组设置里 GPU 仍然是主导项，所以吞吐变化不大（基本在噪声里），但可以看到 worker loop 的一些尾延迟指标在变好：

- `Queue wait p50`: 5.72 → 4.29 ms（-25%）
- `TTFT p50`: 9.21 → 7.60 ms（-17%）

后面如果要把收益放大，可以用更“快”的 decode 路径（比如 `--paged-attn --cuda-graph`）或者更高的持续提交压力，让 CPU overhead 更容易成为瓶颈，再回头看这类改动的收益会更明显。

