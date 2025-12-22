---
classes: wide2
title: "从零实现 LLM Inference：034. Max Active Requests"
excerpt: "pack admission 会把 active sessions 拉得很高：加一个 max_active_requests（max_num_seqs）把 decode backlog 的 ITL/TPOT tail 收回来。"
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

上一版（Prefill Admission Packing）把短请求更快塞进了 prefill budget，TTFT tail 很爽。

但它也会带来一个新的副作用：**active unfinished sessions** 会更快变多，而 decode 每一步只能处理 `max_batch_size` 个 request（round-robin）。

当 active 变得很大时，会出现典型的 decode backlog：

- 单个 request 两个 token 之间要等更久（ITL 变差）
- 每 token 平均耗时拉长（TPOT 变差）

业界通常会有一个类似 `max_num_seqs` 的上限来控制 in-flight 并发。

这次 mini PR 就做一件事：给 `SchedulerManager` 加一个 `max_active_requests`（默认不限制），当 active 已经满了就先 **只 decode 不 admission**，让 decode backlog 收敛。

## 代码变更

### `roseinfer/engine.py`

`OnlineScheduler` 增加一个轻量查询：当前 unfinished 的 session 数。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 04057d7..e14a33b 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1742,6 +1742,9 @@ class OnlineScheduler:
             return True
         return False

+    def num_unfinished(self) -> int:
+        return sum(1 for sess in self._sessions.values() if not sess.finished)
+
     def is_finished(self, request_id: int) -> bool:
         session = self._sessions.get(request_id)
         if session is None:
```

### `roseinfer/server.py`

1. 增加 helper：把本轮 prefill admission 的 `max_reqs` clamp 到剩余 slots。
2. `SchedulerManager` 增加 `max_active_requests` 参数并在 worker loop 生效。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index d4f9885..ae8b0c9 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -110,6 +110,25 @@ class _PendingRequest:

 PrefillAdmissionPolicy = Literal["fifo", "pack"]


+def _cap_prefill_max_reqs(
+    max_reqs: int,
+    *,
+    max_active_requests: Optional[int],
+    active_unfinished: int,
+) -> int:
+    if max_reqs <= 0:
+        raise ValueError("max_reqs must be positive")
+    if active_unfinished < 0:
+        raise ValueError("active_unfinished must be non-negative")
+    if max_active_requests is None:
+        return max_reqs
+    if max_active_requests <= 0:
+        raise ValueError("max_active_requests must be positive")
+    slots = max_active_requests - active_unfinished
+    if slots <= 0:
+        return 0
+    return min(max_reqs, slots)
+
+
 def _take_pending_for_prefill(
@@
 class SchedulerManager:
@@
         prefill_admission_policy: PrefillAdmissionPolicy = "fifo",
         prefill_admission_lookahead: int = 64,
         prefill_force_fifo_every: int = 0,
+        max_active_requests: Optional[int] = None,
     ) -> None:
@@
         self._prefill_iter = 0
+        self._max_active_requests = (
+            int(max_active_requests) if max_active_requests is not None else None
+        )
+        if self._max_active_requests is not None and self._max_active_requests <= 0:
+            raise ValueError("max_active_requests must be positive")
@@
                 with self._lock:
@@
                     force_fifo_every = self._prefill_force_fifo_every
+                    max_active = self._max_active_requests
@@
-                pending = _take_pending_for_prefill(
-                    self._pending_buf,
-                    self._pending,
-                    max_reqs=max_new,
-                    max_tokens=max_tokens,
-                    max_context=max_context,
-                    admission_policy=admission_policy,
-                    lookahead=lookahead,
-                    force_fifo=force_fifo,
-                )
+                admit_cap = _cap_prefill_max_reqs(
+                    max_new,
+                    max_active_requests=max_active,
+                    active_unfinished=self.scheduler.num_unfinished(),
+                )
+                if admit_cap > 0:
+                    pending = _take_pending_for_prefill(
+                        self._pending_buf,
+                        self._pending,
+                        max_reqs=admit_cap,
+                        max_tokens=max_tokens,
+                        max_context=max_context,
+                        admission_policy=admission_policy,
+                        lookahead=lookahead,
+                        force_fifo=force_fifo,
+                    )
+                else:
+                    pending = []
```

### `roseinfer/benchmark_streaming.py`

把 `max_active_requests` 暴露成 CLI，直接做对比。

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
index cfa7439..19f1b2e 100644
--- a/rosellm/roseinfer/benchmark_streaming.py
+++ b/rosellm/roseinfer/benchmark_streaming.py
@@ -126,6 +126,12 @@ def parse_args() -> argparse.Namespace:
         help="Force FIFO admission every N iterations (0: disable).",
     )
+    parser.add_argument(
+        "--max-active-requests",
+        type=int,
+        default=None,
+        help="Max unfinished requests allowed in scheduler (default: unlimited).",
+    )
@@ -254,6 +260,7 @@ def main() -> None:
         prefill_admission_policy=args.prefill_admission_policy,
         prefill_admission_lookahead=int(args.prefill_admission_lookahead),
         prefill_force_fifo_every=int(args.prefill_force_fifo_every),
+        max_active_requests=args.max_active_requests,
         record_token_timestamps=True,
     )
```

### 测试

补一个最小单测覆盖 clamp 逻辑：

```diff
diff --git a/tests/test_max_active_requests.py b/tests/test_max_active_requests.py
new file mode 100644
index 0000000..cf0eb1a
--- /dev/null
+++ b/tests/test_max_active_requests.py
@@ -0,0 +1,43 @@
+import pytest
+
+from rosellm.roseinfer.server import _cap_prefill_max_reqs
+
+
+def test_cap_prefill_max_reqs_unlimited() -> None:
+    assert (
+        _cap_prefill_max_reqs(
+            8,
+            max_active_requests=None,
+            active_unfinished=123,
+        )
+        == 8
+    )
+
+
+def test_cap_prefill_max_reqs_clamps_to_slots() -> None:
+    assert (
+        _cap_prefill_max_reqs(
+            8,
+            max_active_requests=16,
+            active_unfinished=15,
+        )
+        == 1
+    )
+
+
+def test_cap_prefill_max_reqs_returns_zero_when_full() -> None:
+    assert (
+        _cap_prefill_max_reqs(
+            8,
+            max_active_requests=16,
+            active_unfinished=16,
+        )
+        == 0
+    )
+
+
+def test_cap_prefill_max_reqs_validates_args() -> None:
+    with pytest.raises(ValueError, match="max_reqs must be positive"):
+        _cap_prefill_max_reqs(0, max_active_requests=None, active_unfinished=0)
+    with pytest.raises(ValueError, match="active_unfinished must be non-negative"):
+        _cap_prefill_max_reqs(1, max_active_requests=None, active_unfinished=-1)
+    with pytest.raises(ValueError, match="max_active_requests must be positive"):
+        _cap_prefill_max_reqs(1, max_active_requests=0, active_unfinished=0)
```

## 运行

### 单测

```shell
pytest -q
```

输出：

```text
......................                                                   [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
22 passed, 1 warning in 1.69s
```

## Benchmark（HF GPT-2）

这个 workload 让 active sessions 很快堆高，从而放大 decode backlog：

- `--prefill-admission-policy pack`：admission 更激进，短请求更容易进 active set
- decode 还是固定 batch size：`--max-batch-size 8`
- `--prompt-repeats "512,1,1,1"`：长短 prompt mix

### Before（不限制 active）

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cuda \
  --prompt "Hello" \
  --prompt-repeats "512,1,1,1" \
  --unique-prompts \
  --num-requests 128 \
  --submit-interval-ms 0 \
  --max-batch-size 8 \
  --prefill-max-batch-size 128 \
  --prefill-max-tokens 256 \
  --prefill-admission-policy pack \
  --prefill-admission-lookahead 64 \
  --prefill-force-fifo-every 8 \
  --max-new-tokens 32 \
  --no-stop-on-eos \
  --no-prefix-cache
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Requests: 128
Prompt tokens (total): 16864
Completion tokens (total): 4096
Submit wall: 0.105472 s
add_request latency p50/p95/p99: 0.03/0.42/0.43 ms
TTFT p50/p95/p99: 221.31/606.79/684.47 ms
TPOT p50/p95/p99: 154.79/156.46/156.61 ms/token
ITL p50/p95/p99: 152.86/187.09/217.99 ms
Latency p50/p95/p99: 5041.23/5213.39/5219.35 ms
Throughput (completion,total): 769.82 tokens/s
```

### After（限制 active：`--max-active-requests 64`）

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cuda \
  --prompt "Hello" \
  --prompt-repeats "512,1,1,1" \
  --unique-prompts \
  --num-requests 128 \
  --submit-interval-ms 0 \
  --max-batch-size 8 \
  --prefill-max-batch-size 128 \
  --prefill-max-tokens 256 \
  --prefill-admission-policy pack \
  --prefill-admission-lookahead 64 \
  --prefill-force-fifo-every 8 \
  --max-active-requests 64 \
  --max-new-tokens 32 \
  --no-stop-on-eos \
  --no-prefix-cache
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Requests: 128
Prompt tokens (total): 16864
Completion tokens (total): 4096
Submit wall: 0.106439 s
add_request latency p50/p95/p99: 0.03/0.43/0.46 ms
TTFT p50/p95/p99: 1021.92/2334.20/2427.81 ms
TPOT p50/p95/p99: 74.24/100.89/101.35 ms/token
ITL p50/p95/p99: 64.52/103.60/129.61 ms
Latency p50/p95/p99: 3427.21/5275.98/5282.99 ms
Throughput (completion,total): 760.59 tokens/s
```

这组数据里最关键的变化：

- ITL p99：`217.99ms -> 129.61ms`（-40.5%）
- TPOT p99：`156.61 -> 101.35 ms/token`（-35.3%）

同时也能看到一个很正常的取舍：

- `max_active_requests` 本质上是在高负载下把 “decode backlog” 变成 “queueing latency”，所以 TTFT p99 会显著变大
- 这个 knob 需要结合你的目标（更在意 TTFT 还是更在意 ITL/TPOT）去调；默认不限制，保持旧行为

下一步如果想把 TTFT/ITL 同时压住，通常会继续做 chunked prefill / 更细粒度的 token-level budget，让 admission 不至于一次性把很多长 prompt 推进 active set。

