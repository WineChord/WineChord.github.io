---
classes: wide2
title: "从零实现 LLM Inference：032. Decode-first Worker Loop"
excerpt: "worker loop 在有 active sessions 时先 decode 再 prefill：减少 decode 被 admission 抢占，收敛 ITL p99。"
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

前面几版我们把 prefill admission 拆出了独立的 knobs（按 batch size / 按 token budget），已经能把 “prefill 抢占 decode” 这个问题压住不少。

但还有一个很直观的点：worker loop 里 **prefill admission** 和 **decode step** 的执行顺序。

现在的逻辑是：

1. 先把 pending request 拉进来做 admission（可能一次拉很多）
2. 再做 decode step（给已有 session 产出下一 token）

当 pending 很多时，这个顺序会放大两个问题：

- decode 的 token 产出会被 admission 的 CPU/Python 开销推迟，ITL/TPOT tail 会更尖
- GPU 侧也会更容易出现空洞：CPU 先忙一轮 admission，再开始 launch decode kernels

这次 mini PR 做一件小事：给 `SchedulerManager` 增加一个 **decode-first** 的可选策略——当存在 active unfinished sessions 时，每一轮先做一次 decode step，再做 prefill admission（默认不开启，保持旧行为）。

## 代码变更

### `roseinfer/server.py`

新增 `decode_first` 参数，并把 decode 逻辑抽成 `run_decode_once()`：

- `decode_first=False`：保持旧行为（prefill -> decode）
- `decode_first=True`：当 `scheduler.has_unfinished()` 时，decode -> prefill（每轮最多 decode 一次，避免饿死 admission）

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index e2de389..51d9ebe 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -155,6 +155,7 @@ class SchedulerManager:
         max_batch_size: int = 8,
         prefill_max_batch_size: Optional[int] = None,
         prefill_max_tokens: Optional[int] = None,
         record_token_timestamps: bool = False,
+        decode_first: bool = False,
     ) -> None:
         if max_batch_size <= 0:
@@ -168,6 +169,7 @@ class SchedulerManager:
         )
         if self._prefill_max_tokens is not None and self._prefill_max_tokens <= 0:
             raise ValueError("prefill_max_tokens must be positive")
+        self._decode_first = bool(decode_first)

         self.engine = engine
         self.scheduler = OnlineScheduler(
@@ -284,12 +286,56 @@ class SchedulerManager:
     def _worker_loop(self) -> None:
         try:
             while True:
+                def run_decode_once() -> None:
+                    if not self.scheduler.has_unfinished():
+                        return
+                    step_tokens = self.scheduler.step()
+                    finished_ids = self.scheduler.pop_finished_ids()
+
+                    for rid, token_id in step_tokens.items():
+                        with self._lock:
+                            q = self._queues.get(rid)
+                            detok = self._detoks.get(rid)
+                            token_ts = (
+                                self._token_timestamps.get(rid)
+                                if self._record_token_timestamps
+                                else None
+                            )
+                        if q is None or detok is None:
+                            self.scheduler.discard_request(rid)
+                            continue
+                        if token_ts is not None:
+                            token_ts.append(time.perf_counter())
+                        piece = detok.on_token(int(token_id))
+                        if piece:
+                            q.put(piece)
+
+                    for rid in finished_ids:
+                        with self._lock:
+                            q = self._queues.get(rid)
+                            detok = self._detoks.get(rid)
+                        self.scheduler.discard_request(rid)
+                        if q is None:
+                            continue
+                        if detok is not None:
+                            tail = detok.flush()
+                            if tail:
+                                q.put(tail)
+                        q.put(None)
+
                 with self._lock:
                     if not self._running:
                         break
                     max_new = self._prefill_max_batch_size
                     max_tokens = self._prefill_max_tokens
                     max_context = int(self.engine.config.max_position_embeddings)
+                    decode_first = self._decode_first
+
+                did_decode = False
+                if decode_first and self.scheduler.has_unfinished():
+                    run_decode_once()
+                    did_decode = True

                 pending = _take_pending_for_prefill(
                     self._pending_buf,
                     self._pending,
@@ -345,43 +391,8 @@ class SchedulerManager:
                         q.put(None)
                         self.scheduler.discard_request(rid)

-                if self.scheduler.has_unfinished():
-                    step_tokens = self.scheduler.step()
-                    finished_ids = self.scheduler.pop_finished_ids()
-                else:
-                    step_tokens = {}
-                    finished_ids = []
-
-                for rid, token_id in step_tokens.items():
-                    with self._lock:
-                        q = self._queues.get(rid)
-                        detok = self._detoks.get(rid)
-                        token_ts = (
-                            self._token_timestamps.get(rid)
-                            if self._record_token_timestamps
-                            else None
-                        )
-                    if q is None or detok is None:
-                        self.scheduler.discard_request(rid)
-                        continue
-                    if token_ts is not None:
-                        token_ts.append(time.perf_counter())
-                    piece = detok.on_token(int(token_id))
-                    if piece:
-                        q.put(piece)
-
-                for rid in finished_ids:
-                    with self._lock:
-                        q = self._queues.get(rid)
-                        detok = self._detoks.get(rid)
-                    self.scheduler.discard_request(rid)
-                    if q is None:
-                        continue
-                    if detok is not None:
-                        tail = detok.flush()
-                        if tail:
-                            q.put(tail)
-                    q.put(None)
+                if not did_decode:
+                    run_decode_once()

                 if (
                     not pending
                     and not self._pending_buf
                     and not self.scheduler.has_unfinished()
                 ):
                     self._wakeup.wait()
                     self._wakeup.clear()
```

### `roseinfer/benchmark_streaming.py`

加一个 CLI 开关，方便直接对比：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
index 89d2d7e..607bb1f 100644
--- a/rosellm/roseinfer/benchmark_streaming.py
+++ b/rosellm/roseinfer/benchmark_streaming.py
@@ -107,6 +107,12 @@ def parse_args() -> argparse.Namespace:
             "Max prompt tokens to prefill per worker iteration " "(default: unlimited)."
         ),
     )
+    parser.add_argument(
+        "--decode-first",
+        action="store_true",
+        help="Run one decode step before prefill admission when possible.",
+    )
     parser.add_argument(
         "--max-new-tokens",
         type=int,
@@ -226,6 +232,7 @@ def main() -> None:
         max_batch_size=int(args.max_batch_size),
         prefill_max_batch_size=args.prefill_max_batch_size,
         prefill_max_tokens=args.prefill_max_tokens,
+        decode_first=args.decode_first,
         record_token_timestamps=True,
     )
```

### `tests/test_server_streaming.py`

让 streaming 的最小集成测试同时覆盖 `decode_first=False/True`：

```diff
diff --git a/tests/test_server_streaming.py b/tests/test_server_streaming.py
index 1f02f8e..c1d6f62 100644
--- a/tests/test_server_streaming.py
+++ b/tests/test_server_streaming.py
@@ -1,3 +1,4 @@
+import pytest
 import torch

 from rosellm.roseinfer.engine import InferenceEngine
@@
-def test_server_streaming_emits_tokens() -> None:
+@pytest.mark.parametrize("decode_first", [False, True])
+def test_server_streaming_emits_tokens(decode_first: bool) -> None:
@@
-    mgr = SchedulerManager(engine, max_batch_size=2)
+    mgr = SchedulerManager(engine, max_batch_size=2, decode_first=decode_first)
```

## 运行

### 单测

```shell
pytest -q
```

输出：

```text
................                                                         [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
16 passed, 1 warning in 1.65s
```

### Benchmark（HF GPT-2, Streaming）

指标说明（benchmark 脚本里都是用 token timestamps 算出来的）：

- TTFT：`first_token_ts - submit_start`
- ITL：相邻 token 的时间差（所有 token gap 汇总后取 p50/p95/p99）
- TPOT：每个 request 的平均 per-token 时间：`(t_last - t_first)/(n-1)`，再对所有 request 取 p50/p95/p99

这个 workload 的目标是把 admission 压力拉高：

- `--prefill-max-batch-size 128`：每轮 prefill 允许吞很多 pending（放大 admission 的 CPU/Python 开销）
- decode batch size 仍然固定为 8：`--max-batch-size 8`
- `--submit-interval-ms 5`：请求持续到来，保证 decode 和 prefill 在同一条 worker 线程里长期“打架”

#### Before（prefill -> decode）

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cuda \
  --prompt "Hello" \
  --prompt-repeats "1,1,1,64" \
  --unique-prompts \
  --num-requests 128 \
  --submit-interval-ms 5 \
  --max-batch-size 8 \
  --prefill-max-batch-size 128 \
  --max-new-tokens 32 \
  --no-stop-on-eos \
  --no-prefix-cache
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Requests: 128
Prompt tokens (total): 2528
Completion tokens (total): 4096
Submit wall: 0.762151 s
add_request latency p50/p95/p99: 0.09/0.22/0.30 ms
TTFT p50/p95/p99: 13.35/184.04/211.24 ms
TPOT p50/p95/p99: 114.65/117.98/118.04 ms/token
ITL p50/p95/p99: 116.09/127.94/163.72 ms
Latency p50/p95/p99: 3603.14/3668.30/3671.61 ms
Throughput (completion,total): 963.54 tokens/s
```

#### After（decode-first）

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cuda \
  --prompt "Hello" \
  --prompt-repeats "1,1,1,64" \
  --unique-prompts \
  --num-requests 128 \
  --submit-interval-ms 5 \
  --max-batch-size 8 \
  --prefill-max-batch-size 128 \
  --decode-first \
  --max-new-tokens 32 \
  --no-stop-on-eos \
  --no-prefix-cache
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Requests: 128
Prompt tokens (total): 2528
Completion tokens (total): 4096
Submit wall: 0.755462 s
add_request latency p50/p95/p99: 0.07/0.17/0.21 ms
TTFT p50/p95/p99: 13.65/194.52/221.82 ms
TPOT p50/p95/p99: 110.57/112.42/112.51 ms/token
ITL p50/p95/p99: 113.52/118.50/139.56 ms
Latency p50/p95/p99: 3488.86/3561.98/3568.69 ms
Throughput (completion,total): 1016.65 tokens/s
```

这组数据里比较关键的变化：

- ITL p99：`163.72ms -> 139.56ms`（-14.8%）
- TPOT p99：`118.04 -> 112.51 ms/token`（-4.7%）
- Throughput：`963.54 -> 1016.65 tokens/s`（+5.5%）
- 代价是 TTFT p99 有小幅回退：`211.24ms -> 221.82ms`

直觉上这很符合预期：decode-first 让已有 session 的 decode 更不容易被 admission 推迟（尾延迟更稳），但新请求的 prefill 会更晚一点拿到执行机会。

下一步如果要把 TTFT 和 ITL 一起压住，业界通常会继续做 chunked prefill / token-level budget（把 prefill 也切成更小的片段），让 worker loop 能更细粒度地在 prefill 和 decode 之间切换。
