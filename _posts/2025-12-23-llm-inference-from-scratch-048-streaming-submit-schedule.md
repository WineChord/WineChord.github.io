---
classes: wide2
title: "从零实现 LLM Inference：048. Streaming Submit Schedule（absolute vs relative）"
excerpt: "submit_interval_ms 如果用“sleep after each add_request”的相对口径，提交路径一变快，实际到达率就变了，TTFT 会被排队形态污染。给 benchmark_streaming 加 submit_scheudle=absolute，用 t0+i*interval 固定到达节奏，并打印 submit lag。"
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

前面几版我们一直在抠 streaming 相关的指标口径（warmup/repeat、TTFT breakdown、pretok/tokenize workers），但还有一个非常容易踩的坑：

> 你以为自己在固定 `--submit-interval-ms`，其实没有。

当前的实现是：**每次 `add_request()` 结束后 sleep 一段固定时间**。这意味着真实的到达间隔是：

```
实际间隔 = add_request耗时 + submit_interval
```

所以只要你优化了提交路径（比如 `--pretok`、`tokenize_workers`），即使 `--submit-interval-ms` 不变，**实际到达率也会变大**，队列更深，TTFT 可能更差——这时候 TTFT 的变化就不再是“系统变慢”，而是“负载形态变了”。

这一版的目标很明确：给 streaming benchmark 一个“稳定到达率”的口径。

## 代码变更

### `roseinfer/benchmark_streaming.py`

新增一个参数：

- `--submit-schedule {relative,absolute}`
  - `relative`（默认，旧行为）：每次提交后 sleep `submit_interval`
  - `absolute`：第 `i` 个请求目标在 `t0 + i * interval` 触发（提交路径变快不会改变到达节奏）

并在输出里补齐两行：

- `Submit interval/schedule: ...`
- `Submit lag p50/p95/p99`（仅 absolute 且 interval>0 时打印）：实际提交相对目标时间的滞后，用来判断“你设的到达率能不能跑起来”

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
     parser.add_argument(
         "--submit-interval-ms",
         type=float,
         default=0.0,
-        help="Sleep this many milliseconds between request submissions.",
+        help="Submit interval in milliseconds (0: burst).",
     )
+    parser.add_argument(
+        "--submit-schedule",
+        type=str,
+        default="relative",
+        choices=["relative", "absolute"],
+        help=(
+            "How to apply submit interval: "
+            "'relative' sleeps after each submission; "
+            "'absolute' targets t0 + i*interval (less sensitive to add_request overhead)."
+        ),
+    )
@@
 def run_once(...):
     t_global0 = time.perf_counter()
+    submit_interval_s = float(args.submit_interval_ms) / 1e3
+    submit_schedule = str(args.submit_schedule)
+    submit_lags: list[float] = []
     for i, p in enumerate(prompts):
+        if submit_interval_s > 0 and submit_schedule == "absolute":
+            target = t_global0 + float(i) * submit_interval_s
+            if time.perf_counter() < target:
+                time.sleep(target - time.perf_counter())
         submit_start = time.perf_counter()
+        if submit_interval_s > 0 and submit_schedule == "absolute":
+            submit_lags.append(max(0.0, submit_start - target))
         request_id = mgr.add_request(...)
+        if submit_interval_s > 0 and submit_schedule == "relative":
+            time.sleep(submit_interval_s)
@@
     print(f"Submit wall: ...")
+    print(f"Submit interval/schedule: ...")
+    print(f"Submit lag p50/p95/p99: ...")
```

## 运行

```bash
pytest -q
```

```text
..........................                                               [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.11/site-packages/urllib3/util/ssl_.py:260
  /data/projects/rosellm/.conda/lib/python3.11/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
26 passed, 1 warning in 2.15s
```

## Benchmark（HF GPT-2 / streaming）

为了把问题放大，我用一个“prefill 很重但 decode 很轻”的配置：

- `prompt-repeats=256`：让 prompt 很长（tokenization + prefill 都会重）
- `max_new_tokens=1`：decode 几乎不参与
- `submit_interval_ms=1`：对提交路径很敏感

命令模板：

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats 256 --unique-prompts \
  --num-requests 256 --max-new-tokens 1 \
  --max-batch-size 16 --prefill-max-batch-size 16 \
  --submit-interval-ms 1 \
  --no-stop-on-eos --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1
```

### 1) `relative`：提交路径变快 → 到达率变大 → TTFT 被排队形态污染

#### Pretok=off（baseline）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 0
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.375504 s
Submit interval/schedule: 1.000 ms / relative
add_request latency p50/p95/p99: 0.27/0.46/0.55 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 346.06/673.63/722.96 ms
Prefill->first token p50/p95/p99: 69.81/71.01/71.07 ms
TTFT p50/p95/p99: 416.15/743.43/753.41 ms
```

#### Pretok=on（只动提交路径）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Tokenize workers: 0
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.302184 s
Submit interval/schedule: 1.000 ms / relative
add_request latency p50/p95/p99: 0.01/0.07/0.08 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 359.01/757.14/768.85 ms
Prefill->first token p50/p95/p99: 69.87/70.75/71.00 ms
TTFT p50/p95/p99: 428.94/817.34/828.71 ms
```

可以看到：

- `Submit wall` 变短（提交更快）
- 但 `Queue wait/TTFT` 反而变长（到达率变大、排队更深）

这不是系统算子变慢，是 benchmark 的到达节奏变了。

### 2) `absolute`：到达节奏固定，TTFT 对比更干净

#### Pretok=off

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 0
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.255373 s
Submit interval/schedule: 1.000 ms / absolute
Submit lag p50/p95/p99: 0.05/0.07/0.36 ms
add_request latency p50/p95/p99: 0.24/0.30/0.40 ms
Queue wait (post-tok) p50/p95/p99: 391.46/772.76/827.47 ms
TTFT p50/p95/p99: 461.04/843.42/868.08 ms
```

#### Pretok=on

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Tokenize workers: 0
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.255156 s
Submit interval/schedule: 1.000 ms / absolute
Submit lag p50/p95/p99: 0.06/0.08/0.12 ms
add_request latency p50/p95/p99: 0.02/0.07/0.10 ms
Queue wait (post-tok) p50/p95/p99: 394.67/776.21/831.38 ms
TTFT p50/p95/p99: 464.23/845.39/866.53 ms
```

这里 `Submit wall` 和 `TTFT p99` 基本对齐（差异在噪声范围内），说明：

- 到达节奏确实被固定住了
- `--pretok` 只在“提交路径”生效，不再通过改变到达率间接影响 TTFT

## 结论

这版不是性能优化，它是 benchmark 口径修复：

- `relative` 更像“客户端串行发请求”的真实行为（上一个请求提交完再 sleep）
- `absolute` 更适合做 **前后对比**：只要你想隔离“系统变了 vs 负载形态变了”，就用它

实践建议：

- 做对比时用 `--submit-schedule absolute`，并盯一下 `Submit lag`：
  - lag 很小：说明你设的到达率能跑起来，对比可信
  - lag 很大：说明你已经发不动了（到达率太高或提交太慢），这时 TTFT/排队的意义也会变
