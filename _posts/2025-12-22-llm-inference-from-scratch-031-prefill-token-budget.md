---
classes: wide2
title: "从零实现 LLM Inference：031. Prefill Token Budget"
excerpt: "prefill admission 从“按请求数”升级为“按 tokens 预算”：限制 prefill 抢占，收敛 ITL p99。"
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

上一版（Prefill Admission Knob）把 **prefill admission** 和 **decode batch size** 解耦了：prefill 可以一次吞更多 pending，而 decode 仍然保持自己的 batch size。

但只用 “按请求数（max batch size）” 依然不够精细：当 prompt 长短差异很大时，worker 可能在某一轮 admission 里把大量 prompt tokens 一口气塞进 prefill，导致：

- prefill 时间变长，decode 这轮被迫等着（ITL/TPOT tail 会出现尖峰）
- 这类问题本质上是 **prefill 抢占 decode**，需要一个更接近 compute 的预算口径

业界常见的做法是引入 token budget（比如 vLLM 的 `max_num_batched_tokens` 思路）：限制每轮 prefill 允许吃掉的 tokens 上限。

这次 mini PR 就做一件事：给 `SchedulerManager` 加一个 **prefill token budget**。

## 代码变更

### `roseinfer/server.py`

新增 `prefill_max_tokens`：

- `None` 表示不限制（保持旧行为）
- admission 时按 FIFO 取 pending request，累计 `min(len(prompt_token_ids), max_context)` 作为 cost
- 如果下一个 request 会超预算：留在 buffer 里，下一轮再处理（保证顺序不乱）
- 如果第一个 request 本身就超预算：允许它单独成 batch（否则会卡死）

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 42aedd2..007bb93 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -1,4 +1,5 @@
 import argparse
+from collections import deque
 import queue
 import threading
 import time
@@
 class _PendingRequest:
@@
     do_sample: bool
 
 
+def _take_pending_for_prefill(
+    pending_buf: "deque[_PendingRequest]",
+    pending_q: "queue.Queue[_PendingRequest]",
+    *,
+    max_reqs: int,
+    max_tokens: Optional[int],
+    max_context: int,
+) -> list[_PendingRequest]:
+    if max_reqs <= 0:
+        raise ValueError("max_reqs must be positive")
+    if max_context <= 0:
+        raise ValueError("max_context must be positive")
+    if max_tokens is not None and max_tokens <= 0:
+        raise ValueError("max_tokens must be positive")
+
+    out: list[_PendingRequest] = []
+    tokens_used = 0
+    while len(out) < max_reqs:
+        if pending_buf:
+            req = pending_buf.popleft()
+        else:
+            try:
+                req = pending_q.get_nowait()
+            except queue.Empty:
+                break
+
+        cost = min(len(req.prompt_token_ids), max_context)
+        if max_tokens is not None:
+            if not out and cost > max_tokens:
+                out.append(req)
+                break
+            if out and tokens_used + cost > max_tokens:
+                pending_buf.appendleft(req)
+                break
+
+        out.append(req)
+        tokens_used += cost
+    return out
+
 class SchedulerManager:
     def __init__(
         self,
         engine: InferenceEngine,
         max_batch_size: int = 8,
         prefill_max_batch_size: Optional[int] = None,
+        prefill_max_tokens: Optional[int] = None,
         record_token_timestamps: bool = False,
     ) -> None:
@@
         self._prefill_max_batch_size = int(prefill_max_batch_size)
@@
+        self._prefill_max_tokens = (
+            int(prefill_max_tokens) if prefill_max_tokens is not None else None
+        )
+        if self._prefill_max_tokens is not None and self._prefill_max_tokens <= 0:
+            raise ValueError("prefill_max_tokens must be positive")
@@
         self._pending: "queue.Queue[_PendingRequest]" = queue.Queue()
+        self._pending_buf: "deque[_PendingRequest]" = deque()
@@
                 with self._lock:
                     if not self._running:
                         break
                     max_new = self._prefill_max_batch_size
+                    max_tokens = self._prefill_max_tokens
+                    max_context = int(self.engine.config.max_position_embeddings)
-                pending: list[_PendingRequest] = []
-                for _ in range(max_new):
-                    try:
-                        pending.append(self._pending.get_nowait())
-                    except queue.Empty:
-                        break
+                pending = _take_pending_for_prefill(
+                    self._pending_buf,
+                    self._pending,
+                    max_reqs=max_new,
+                    max_tokens=max_tokens,
+                    max_context=max_context,
+                )
@@
-                if (
-                    not pending
-                    and not self._pending_buf
-                    and not self.scheduler.has_unfinished()
-                ):
+                if (
+                    not pending
+                    and not self._pending_buf
+                    and not self.scheduler.has_unfinished()
+                ):
                     self._wakeup.wait()
                     self._wakeup.clear()
```

### `roseinfer/benchmark_streaming.py`

为了能稳定复现 “prefill 抢占 decode”：

1. 新增 `--submit-interval-ms`：模拟持续到来的请求（不再是纯 burst）。
2. 新增 `--prompt-repeats`：做一个短/长 prompt mix（长 prompt 用重复 base prompt 来构造）。
3. 新增 `--prefill-max-tokens`：把 token budget 暴露成 CLI。

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
index f9c7fcc..39a5a1d 100644
--- a/rosellm/roseinfer/benchmark_streaming.py
+++ b/rosellm/roseinfer/benchmark_streaming.py
@@ -44,6 +44,14 @@ def parse_args() -> argparse.Namespace:
     parser.add_argument(
         "--prompt",
         type=str,
         required=True,
         help="Prompt to generate text from",
     )
+    parser.add_argument(
+        "--prompt-repeats",
+        type=str,
+        default=None,
+        help=(
+            "Comma-separated repeat counts for base prompt, cycled per request. "
+            'Example: \"1,1,1,64\" produces a 3-short+1-long mix.'
+        ),
+    )
@@
     parser.add_argument(
         "--num-requests",
         type=int,
         default=16,
         help="Number of streaming requests",
     )
+    parser.add_argument(
+        "--submit-interval-ms",
+        type=float,
+        default=0.0,
+        help="Sleep this many milliseconds between request submissions.",
+    )
@@
     parser.add_argument(
         "--prefill-max-batch-size",
         type=int,
         default=None,
         help=(
             "Max pending requests to prefill per worker iteration "
             "(default: same as --max-batch-size)."
         ),
     )
+    parser.add_argument(
+        "--prefill-max-tokens",
+        type=int,
+        default=None,
+        help=(
+            "Max prompt tokens to prefill per worker iteration "
+            "(default: unlimited)."
+        ),
+    )
@@
     prompts: list[str] = []
     for i in range(int(args.num_requests)):
         rep = repeats[i % len(repeats)] if repeats is not None else 1
         p = " ".join([args.prompt] * rep) if rep > 1 else args.prompt
         if args.unique_prompts:
             p = f"{p} [{i}]"
         prompts.append(p)
@@
     mgr = SchedulerManager(
         engine,
         max_batch_size=int(args.max_batch_size),
         prefill_max_batch_size=args.prefill_max_batch_size,
+        prefill_max_tokens=args.prefill_max_tokens,
         record_token_timestamps=True,
     )
@@
             th.start()
+            if args.submit_interval_ms > 0:
+                time.sleep(float(args.submit_interval_ms) / 1e3)
```

### 新增测试

主要测 `_take_pending_for_prefill` 的边界条件（预算截断 / 单条 oversize / max_context 截断）：

```diff
diff --git a/tests/test_scheduler_manager_prefill_max_tokens.py b/tests/test_scheduler_manager_prefill_max_tokens.py
new file mode 100644
index 0000000..131a72f
--- /dev/null
+++ b/tests/test_scheduler_manager_prefill_max_tokens.py
@@ -0,0 +1,76 @@
+import queue
+from collections import deque
+
+import pytest
+
+from rosellm.roseinfer.server import _PendingRequest, _take_pending_for_prefill
+
+def _req(rid: int, n: int) -> _PendingRequest:
+    return _PendingRequest(
+        request_id=int(rid),
+        prompt="",
+        prompt_token_ids=[1] * int(n),
+        max_new_tokens=1,
+        temperature=1.0,
+        top_k=0,
+        top_p=1.0,
+        stop_on_eos=False,
+        do_sample=False,
+    )
+
+
+def test_take_pending_for_prefill_respects_max_tokens_fifo() -> None:
+    buf: deque[_PendingRequest] = deque()
+    q: "queue.Queue[_PendingRequest]" = queue.Queue()
+    q.put(_req(0, 2))
+    q.put(_req(1, 2))
+    q.put(_req(2, 2))
+
+    out = _take_pending_for_prefill(
+        buf,
+        q,
+        max_reqs=8,
+        max_tokens=4,
+        max_context=1024,
+    )
+    assert [r.request_id for r in out] == [0, 1]
+    assert list(buf)[0].request_id == 2
+
+
+def test_take_pending_for_prefill_allows_single_oversize_request() -> None:
+    buf: deque[_PendingRequest] = deque()
+    q: "queue.Queue[_PendingRequest]" = queue.Queue()
+    q.put(_req(0, 100))
+    q.put(_req(1, 1))
+
+    out = _take_pending_for_prefill(
+        buf,
+        q,
+        max_reqs=8,
+        max_tokens=4,
+        max_context=1024,
+    )
+    assert [r.request_id for r in out] == [0]
+    assert q.get_nowait().request_id == 1
+
+
+def test_take_pending_for_prefill_uses_max_context_for_cost() -> None:
+    buf: deque[_PendingRequest] = deque()
+    q: "queue.Queue[_PendingRequest]" = queue.Queue()
+    q.put(_req(0, 100))
+    q.put(_req(1, 100))
+
+    out = _take_pending_for_prefill(
+        buf,
+        q,
+        max_reqs=8,
+        max_tokens=16,
+        max_context=8,
+    )
+    assert [r.request_id for r in out] == [0, 1]
+
+
+def test_take_pending_for_prefill_validates_args() -> None:
+    buf: deque[_PendingRequest] = deque()
+    q: "queue.Queue[_PendingRequest]" = queue.Queue()
+    with pytest.raises(ValueError, match="max_tokens must be positive"):
+        _take_pending_for_prefill(buf, q, max_reqs=1, max_tokens=0, max_context=1)
```

## 运行

### 单测

```shell
pytest -q
```

输出：

```text
...............                                                          [100%]
15 passed, 1 warning in 1.63s
```

### Benchmark（HF GPT-2）

这个 workload 是刻意挑出来的：

- `--submit-interval-ms 20`：请求不是一次性 burst 全灌进来
- `--prompt-repeats "1,1,1,64"`：3 个短 prompt + 1 个长 prompt 循环
- decode batch size 固定为 8（我们只想看 prefill admission 对 ITL 的影响）

#### Before（不限制 prefill tokens）

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --prompt-repeats "1,1,1,64" \
  --unique-prompts \
  --num-requests 32 \
  --submit-interval-ms 20 \
  --max-batch-size 8 \
  --prefill-max-batch-size 32 \
  --max-new-tokens 32 \
  --no-stop-on-eos
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 632
Completion tokens (total): 1024
Submit wall: 0.745182 s
add_request latency p50/p95/p99: 0.27/1.20/53.85 ms
TTFT p50/p95/p99: 318.24/506.89/532.18 ms
TPOT p50/p95/p99: 153.08/157.92/159.20 ms/token
ITL p50/p95/p99: 147.31/175.81/438.15 ms
Latency p50/p95/p99: 4960.68/5096.98/5108.77 ms
Throughput (completion,total): 186.22 tokens/s
```

#### After（限制每轮 prefill tokens：`--prefill-max-tokens 224`）

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --prompt-repeats "1,1,1,64" \
  --unique-prompts \
  --num-requests 32 \
  --submit-interval-ms 20 \
  --max-batch-size 8 \
  --prefill-max-batch-size 32 \
  --prefill-max-tokens 224 \
  --max-new-tokens 32 \
  --no-stop-on-eos
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 632
Completion tokens (total): 1024
Submit wall: 0.747317 s
add_request latency p50/p95/p99: 0.28/1.27/54.70 ms
TTFT p50/p95/p99: 309.82/441.69/463.65 ms
TPOT p50/p95/p99: 149.61/154.38/156.88 ms/token
ITL p50/p95/p99: 143.34/159.77/340.15 ms
Latency p50/p95/p99: 4867.80/5011.03/5024.45 ms
Throughput (completion,total): 189.24 tokens/s
```

这组数据说明两件事：

- **prefill token budget 能收敛 ITL p99**：`438.15ms -> 340.15ms`（~1.29x）
- 同时不会把 TTFT/吞吐/总延迟弄崩（这组参数下反而都更好）

下一步如果要继续把尾延迟打下去，就需要更像业界的做法：用 token budget 统一约束 “prefill + decode” 的整批 tokens，甚至做 chunked prefill（避免长 prompt 把 decode 打断太久）。
