---
classes: wide2
title: "从零实现 LLM Inference：033. Prefill Admission Packing"
excerpt: "token budget + FIFO 仍然会遇到 head-of-line blocking：用 lookahead packing 把短请求先塞进 prefill，收敛 TTFT p99。"
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

上一版我们引入了 **prefill token budget**：限制每轮 admission 允许吃掉的 prompt tokens，从而避免一轮 prefill 抢占太久。

但只要 admission 还是 FIFO，prompt 长短差异很大时，依然会遇到典型的 **head-of-line blocking**：

- 队头是长 prompt（甚至超过 budget）：这一轮只能先处理它（或被迫单独成 batch），后面的短 prompt 明明很快，却要一起排队等
- 结果就是：短请求的 **TTFT tail** 被长请求“拖着走”

业界常见做法是在 token budget 的基础上再做一步：**在 waiting queue 上做 packing**（简单说就是“看一眼后面的，先把能塞进去的小的塞满”）。

这次 mini PR 就做这件事：

- 增加 `prefill_admission_policy=fifo|pack`（默认 fifo，不破坏旧行为）
- `pack` 模式下对 pending 做一个 lookahead window，按 cost（prompt tokens）greedy 选一组塞进 budget
- 加一个公平性兜底：`prefill_force_fifo_every`，每 N 轮强制 FIFO，避免长 prompt 永远被跳过

## 代码变更

### `roseinfer/server.py`

- 新增 admission policy + lookahead + force FIFO knobs
- `_take_pending_for_prefill` 支持 `pack`：在 window 内按 token cost greedy 选取（输出顺序仍按 arrival 保持稳定）

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 0f28643..d4f9885 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -107,6 +107,9 @@ class _PendingRequest:
     do_sample: bool


+PrefillAdmissionPolicy = Literal["fifo", "pack"]
+
+
 def _take_pending_for_prefill(
     pending_buf: "deque[_PendingRequest]",
     pending_q: "queue.Queue[_PendingRequest]",
@@ -114,6 +117,9 @@ def _take_pending_for_prefill(
     max_reqs: int,
     max_tokens: Optional[int],
     max_context: int,
+    admission_policy: PrefillAdmissionPolicy,
+    lookahead: int,
+    force_fifo: bool,
 ) -> list[_PendingRequest]:
     if max_reqs <= 0:
         raise ValueError("max_reqs must be positive")
@@ -121,29 +127,75 @@ def _take_pending_for_prefill(
         raise ValueError("max_context must be positive")
     if max_tokens is not None and max_tokens <= 0:
         raise ValueError("max_tokens must be positive")
+    if lookahead <= 0:
+        raise ValueError("lookahead must be positive")
+
+    if force_fifo or admission_policy == "fifo" or max_tokens is None:
+        out: list[_PendingRequest] = []
+        tokens_used = 0
+        while len(out) < max_reqs:
+            if pending_buf:
+                req = pending_buf.popleft()
+            else:
+                try:
+                    req = pending_q.get_nowait()
+                except queue.Empty:
+                    break
+
+            cost = min(len(req.prompt_token_ids), max_context)
+            if max_tokens is not None:
+                if not out and cost > max_tokens:
+                    out.append(req)
+                    break
+                if out and tokens_used + cost > max_tokens:
+                    pending_buf.appendleft(req)
+                    break
+
+            out.append(req)
+            tokens_used += cost
+        return out
+
+    # admission_policy == "pack" and max_tokens is not None.
+    window: list[_PendingRequest] = []
+    while len(window) < lookahead:
+        if pending_buf:
+            window.append(pending_buf.popleft())
+            continue
+        try:
+            window.append(pending_q.get_nowait())
+        except queue.Empty:
+            break
+    if not window:
+        return []
+
+    costs = [min(len(req.prompt_token_ids), max_context) for req in window]
+    order = sorted(range(len(window)), key=lambda i: (costs[i], i))
+    selected = [False for _ in window]
+    tokens_used = 0
+    selected_count = 0
+    for idx in order:
+        if selected_count >= max_reqs:
+            break
+        cost = costs[idx]
+        if cost > max_tokens:
+            continue
+        if tokens_used + cost > max_tokens:
+            continue
+        selected[idx] = True
+        tokens_used += cost
+        selected_count += 1
+    if selected_count == 0:
+        selected[0] = True
+
+    out: list[_PendingRequest] = []
+    for idx, req in enumerate(window):
+        if selected[idx]:
+            out.append(req)
+            if len(out) >= max_reqs:
+                break
+    for idx in range(len(window) - 1, -1, -1):
+        if not selected[idx]:
+            pending_buf.appendleft(window[idx])
     return out


@@ -156,6 +208,9 @@ class SchedulerManager:
         prefill_max_tokens: Optional[int] = None,
         record_token_timestamps: bool = False,
         decode_first: bool = False,
+        prefill_admission_policy: PrefillAdmissionPolicy = "fifo",
+        prefill_admission_lookahead: int = 64,
+        prefill_force_fifo_every: int = 0,
     ) -> None:
         if max_batch_size <= 0:
             raise ValueError("max_batch_size must be positive")
@@ -170,6 +225,16 @@ class SchedulerManager:
         if self._prefill_max_tokens is not None and self._prefill_max_tokens <= 0:
             raise ValueError("prefill_max_tokens must be positive")
         self._decode_first = bool(decode_first)
+        if prefill_admission_policy not in ("fifo", "pack"):
+            raise ValueError("prefill_admission_policy must be fifo|pack")
+        self._prefill_admission_policy = prefill_admission_policy
+        self._prefill_admission_lookahead = int(prefill_admission_lookahead)
+        if self._prefill_admission_lookahead <= 0:
+            raise ValueError("prefill_admission_lookahead must be positive")
+        self._prefill_force_fifo_every = int(prefill_force_fifo_every)
+        if self._prefill_force_fifo_every < 0:
+            raise ValueError("prefill_force_fifo_every must be non-negative")
+        self._prefill_iter = 0
@@
                 with self._lock:
                     if not self._running:
                         break
                     max_new = self._prefill_max_batch_size
                     max_tokens = self._prefill_max_tokens
                     max_context = int(self.engine.config.max_position_embeddings)
                     decode_first = self._decode_first
+                    admission_policy = self._prefill_admission_policy
+                    lookahead = self._prefill_admission_lookahead
+                    force_fifo_every = self._prefill_force_fifo_every
+
+                self._prefill_iter += 1
+                force_fifo = (
+                    force_fifo_every > 0
+                    and (self._prefill_iter % force_fifo_every == 0)
+                )
@@
                 pending = _take_pending_for_prefill(
                     self._pending_buf,
                     self._pending,
                     max_reqs=max_new,
                     max_tokens=max_tokens,
                     max_context=max_context,
+                    admission_policy=admission_policy,
+                    lookahead=lookahead,
+                    force_fifo=force_fifo,
                 )
```

### `roseinfer/benchmark_streaming.py`

把 knobs 暴露成 CLI，方便做 before/after：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
index a06d549..cfa7439 100644
--- a/rosellm/roseinfer/benchmark_streaming.py
+++ b/rosellm/roseinfer/benchmark_streaming.py
@@ -107,6 +107,25 @@ def parse_args() -> argparse.Namespace:
             "Max prompt tokens to prefill per worker iteration " "(default: unlimited)."
         ),
     )
+    parser.add_argument(
+        "--prefill-admission-policy",
+        type=str,
+        default="fifo",
+        choices=["fifo", "pack"],
+        help="Prefill admission policy (default: fifo).",
+    )
+    parser.add_argument(
+        "--prefill-admission-lookahead",
+        type=int,
+        default=64,
+        help="Pending lookahead window for pack admission.",
+    )
+    parser.add_argument(
+        "--prefill-force-fifo-every",
+        type=int,
+        default=0,
+        help="Force FIFO admission every N iterations (0: disable).",
+    )
@@ -231,6 +250,9 @@ def main() -> None:
         prefill_max_batch_size=args.prefill_max_batch_size,
         prefill_max_tokens=args.prefill_max_tokens,
         decode_first=args.decode_first,
+        prefill_admission_policy=args.prefill_admission_policy,
+        prefill_admission_lookahead=int(args.prefill_admission_lookahead),
+        prefill_force_fifo_every=int(args.prefill_force_fifo_every),
         record_token_timestamps=True,
     )
```

### 测试

补一个 packing 的核心语义测试：能绕过 oversize 队头，把短请求先塞进去，同时保证不会卡死。

```diff
diff --git a/tests/test_prefill_admission_packing.py b/tests/test_prefill_admission_packing.py
new file mode 100644
index 0000000..2836c43
--- /dev/null
+++ b/tests/test_prefill_admission_packing.py
@@ -0,0 +1,66 @@
+import queue
+from collections import deque
+
+from rosellm.roseinfer.server import _PendingRequest, _take_pending_for_prefill
+
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
+def test_take_pending_for_prefill_pack_skips_oversize_head() -> None:
+    buf: deque[_PendingRequest] = deque()
+    q: "queue.Queue[_PendingRequest]" = queue.Queue()
+    q.put(_req(0, 100))
+    q.put(_req(1, 2))
+    q.put(_req(2, 2))
+
+    out = _take_pending_for_prefill(
+        buf,
+        q,
+        max_reqs=8,
+        max_tokens=4,
+        max_context=1024,
+        admission_policy="pack",
+        lookahead=16,
+        force_fifo=False,
+    )
+    assert [r.request_id for r in out] == [1, 2]
+    assert list(buf)[0].request_id == 0
+    try:
+        q.get_nowait()
+    except queue.Empty:
+        pass
+    else:
+        raise AssertionError("expected queue to be empty")
+
+
+def test_take_pending_for_prefill_pack_progresses_when_all_oversize() -> None:
+    buf: deque[_PendingRequest] = deque()
+    q: "queue.Queue[_PendingRequest]" = queue.Queue()
+    q.put(_req(0, 100))
+    q.put(_req(1, 100))
+
+    out = _take_pending_for_prefill(
+        buf,
+        q,
+        max_reqs=8,
+        max_tokens=4,
+        max_context=1024,
+        admission_policy="pack",
+        lookahead=16,
+        force_fifo=False,
+    )
+    assert [r.request_id for r in out] == [0]
+    assert list(buf)[0].request_id == 1
```

（另外，把 `test_scheduler_manager_prefill_max_tokens.py` 里 `_take_pending_for_prefill` 的调用签名同步了一下，这里就不重复贴了。）

## 运行

### 单测

```shell
pytest -q
```

输出：

```text
..................                                                       [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
18 passed, 1 warning in 1.63s
```

## Benchmark（HF GPT-2）

这个 workload 刻意制造 HOL：

- `--prefill-max-tokens 256`：每轮 admission 有 token budget
- `--prompt-repeats "512,1,1,1"`：队列里持续出现 oversize 长 prompt
- FIFO 下：每遇到一个 oversize 队头，就会把后面的短请求一起“拖住”
- pack 下：lookahead 里先把能塞进 budget 的短请求取出来，长请求留在队头（并用 `--prefill-force-fifo-every 8` 保证它也能被处理）

### Before（FIFO）

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
Submit wall: 0.103464 s
add_request latency p50/p95/p99: 0.03/0.42/0.46 ms
TTFT p50/p95/p99: 672.29/1106.39/1144.49 ms
TPOT p50/p95/p99: 142.26/145.73/146.08 ms/token
ITL p50/p95/p99: 152.93/160.77/189.82 ms
Latency p50/p95/p99: 5184.35/5341.20/5347.30 ms
Throughput (completion,total): 751.55 tokens/s
```

### After（Pack + force FIFO）

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
Submit wall: 0.105095 s
add_request latency p50/p95/p99: 0.03/0.42/0.46 ms
TTFT p50/p95/p99: 216.77/611.07/689.69 ms
TPOT p50/p95/p99: 156.40/158.05/158.17 ms/token
ITL p50/p95/p99: 155.06/190.19/221.63 ms
Latency p50/p95/p99: 5086.20/5258.17/5263.87 ms
Throughput (completion,total): 763.48 tokens/s
```

这组数据里最核心的变化：

- TTFT p99：`1144.49ms -> 689.69ms`（-39.7%）
- Latency p99：`5347.30ms -> 5263.87ms`（-1.6%）
- Throughput：`751.55 -> 763.48 tokens/s`（+1.6%）

同时也能看到一个典型取舍：

- pack 会让更多短请求更早进入 active set，decode 侧的并发更快变高，所以 TPOT/ITL 的 tail 可能会上升
- 这部分一般需要配合后续的策略一起做：比如更细粒度的 admission（chunked prefill）、或者把 decode 的公平性/优先级再细化

