---
classes: wide2
title: "从零实现 LLM Inference：049. Max Inflight Requests（过载保护 / 429）"
excerpt: "burst streaming 下 unlimited inflight 会把排队延迟炸到 p99，但吞吐几乎不变。给 SchedulerManager 加 max_inflight_requests：超过就直接拒绝（HTTP 429），benchmark_streaming 也打印 Rejected 与 reject latency。"
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

前面我们已经有了 `max_active_requests`（限制 scheduler 内部“正在跑”的请求数），但它只解决了“活跃集合”的上限问题：如果入口一直把请求塞进来，pending queue 依然可以无限增长，最终就会出现：

- GPU 吞吐没变（甚至还更稳定）
- 但排队更深，`Queue wait / TTFT` 的 p99 被无限放大

这一版引入一个更贴近服务语义的 knob：**max inflight requests**。超过上限就直接拒绝（流式接口返回 `HTTP 429`），把系统从“无限排队”改成“有限排队 + 明确回压”。

## 代码变更

### `roseinfer/server.py`

`SchedulerManager` 新增：

- `max_inflight_requests: Optional[int] = None`
  - `None`：不限制（旧行为）
  - `>0`：`add_request()` 在创建 request_id 前先做 inflight 上限判断，满了直接抛 `SchedulerManagerOverloadedError`

FastAPI 流式接口增加一个很薄的转换层：捕获 `SchedulerManagerOverloadedError`，返回 `HTTP 429`。

同时 server CLI 暴露 `--max-inflight-requests`，让这个策略可以直接在服务进程里打开。

核心 diff：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
-from fastapi import FastAPI
+from fastapi import FastAPI, HTTPException
+
+class SchedulerManagerOverloadedError(RuntimeError):
+    pass
@@
 class SchedulerManager:
     def __init__(..., max_active_requests: Optional[int] = None, ...):
+        self._max_inflight_requests = (
+            int(max_inflight_requests) if max_inflight_requests is not None else None
+        )
@@
     def add_request(...):
         with self._lock:
+            if self._max_inflight_requests is not None and (
+                len(self._queues) >= self._max_inflight_requests
+            ):
+                raise SchedulerManagerOverloadedError("too many inflight requests")
@@
-def create_app(engine: InferenceEngine) -> FastAPI:
+def create_app(engine: InferenceEngine, *, max_inflight_requests: Optional[int] = None) -> FastAPI:
     app = FastAPI(...)
-    sched_manager = SchedulerManager(engine, max_batch_size=8)
+    sched_manager = SchedulerManager(engine, max_batch_size=8, max_inflight_requests=max_inflight_requests)
@@
     if body.stream:
-        request_id = sched_manager.add_request(...)
+        try:
+            request_id = sched_manager.add_request(...)
+        except SchedulerManagerOverloadedError as exc:
+            raise HTTPException(status_code=429, detail=str(exc)) from exc
@@
 parser.add_argument("--max-inflight-requests", type=int, default=None, ...)
 app = create_app(engine, max_inflight_requests=args.max_inflight_requests)
```

### `roseinfer/benchmark_streaming.py`

新增一个对齐 server 的参数：

- `--max-inflight-requests`

并在 burst submit 时：

- `add_request()` 抛 overload → 计入 `Rejected`，记录一次 `reject latency`
- 输出里增加：
  - `Rejected: N`
  - `reject latency p50/p95/p99`

另外，这里顺手修了一个“在有 reject 时容易误导”的口径：`Prompt tokens (total)` 改成只统计**被接受**的请求（否则会出现 `Requests=64` 但 prompt tokens 还是 256 个请求的总和）。

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
+parser.add_argument("--max-inflight-requests", type=int, default=None, ...)
@@
 mgr = SchedulerManager(..., max_inflight_requests=args.max_inflight_requests, ...)
@@
+rejected_lats: list[float] = []
+accepted_prompt_lens: list[int] = []
 try:
     request_id = mgr.add_request(...)
 except SchedulerManagerOverloadedError:
     rejected_lats.append(time.perf_counter() - submit_start)
     continue
+accepted_prompt_lens.append(prompt_len)
@@
 print(f"Requests: {len(results)}")
+print(f"Rejected: {len(rejected_lats)}")
-print(f"Prompt tokens (total): {sum(prompt_lens)}")
+print(f"Prompt tokens (total): {sum(accepted_prompt_lens)}")
```

### `tests/test_max_inflight_requests.py`

补一个最小语义测试：

- `max_inflight_requests=1` 时，第二个请求必然被拒绝
- 消费完第一个 stream 后，再次 `add_request()` 能成功

## 运行

```bash
pytest -q
```

```text
...........................                                              [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
27 passed, 1 warning in 2.12s
```

## Benchmark（HF GPT-2 / streaming）

为了把“排队延迟被 inflight 放大”这个现象放大，我用一个典型配置：

- `prompt-repeats=259`：prefill 重
- `max_new_tokens=1`：decode 极轻（TPOT/ITL 基本不参与）
- `--no-prefix-cache`：避免 prefix cache 形态干扰
- `--tokenize-workers 4`：提交路径更接近“轻量 enqueue”
- `submit_interval_ms=0`：burst 提交，最容易把 pending 拉深

### 1) baseline：无限 inflight

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats 259 \
  --num-requests 256 --max-new-tokens 1 \
  --tokenize-workers 4 \
  --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1
```

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 4
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.030277 s
add_request latency p50/p95/p99: 0.01/0.02/0.05 ms
Tokenize p50/p95/p99: 0.31/0.55/0.76 ms
Queue wait (post-tok) p50/p95/p99: 452.28/871.98/900.39 ms
Prefill->first token p50/p95/p99: 28.82/29.04/29.26 ms
TTFT p50/p95/p99: 481.51/901.06/921.20 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 481.90/901.25/921.30 ms
Throughput (completion,total): 269.17 tokens/s
```

### 2) 开启回压：`--max-inflight-requests 64`

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats 259 \
  --num-requests 256 --max-new-tokens 1 \
  --tokenize-workers 4 \
  --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1 \
  --max-inflight-requests 64
```

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 4
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Rejected: 192
Prompt tokens (total): 16576
Completion tokens (total): 64
Submit wall: 0.007906 s
add_request latency p50/p95/p99: 0.01/0.01/0.04 ms
reject latency p50/p95/p99: 0.00/0.00/0.00 ms
Tokenize p50/p95/p99: 0.39/0.68/0.85 ms
Queue wait (post-tok) p50/p95/p99: 93.78/206.03/206.39 ms
Prefill->first token p50/p95/p99: 28.89/28.99/29.00 ms
TTFT p50/p95/p99: 123.04/231.08/231.31 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 123.45/231.25/231.50 ms
Throughput (completion,total): 268.05 tokens/s
```

### 结论

在这个“prefill 重 / decode 轻 / burst submit”的场景里：

- **吞吐几乎不变**（269 → 268 tokens/s）：瓶颈在 prefill 的 GPU 计算，inflight 拉得再深也不会让 GPU 更快
- 但 **tail latency 被显著收敛**：
  - `Queue wait p99`: 900ms → 206ms
  - `TTFT p99`: 921ms → 231ms
- 代价是：超过上限的请求会被快速拒绝（这里 `Rejected: 192`）

这就是服务里常见的取舍：**用明确的回压/拒绝，换 tail latency 可控**。后面再往上可以继续加：重试/退避策略、按 token budget 的 admission、优先级队列等，把“拒绝”做得更聪明。

