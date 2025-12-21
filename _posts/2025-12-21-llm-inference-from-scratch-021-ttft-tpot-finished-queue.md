---
classes: wide2
title: "从零实现 LLM Inference：021. TTFT/TPOT + Finished Queue"
excerpt: "给 benchmark 加上 TTFT/TPOT，并把 server 的 finished 清理从扫描改成事件。"
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

在前面的 benchmark 里，我们主要看 tokens/s，但做推理引擎其实还需要两个更能反映体验的指标：

1. TTFT（time to first token）：从请求进入到拿到第一个 token 的时间。
2. TPOT（time per output token）：decode 阶段平均每个 token 的耗时。

另外在 server 的 streaming worker 里，我之前偷懒用了一个 O(n) 的扫描：每步 decode 完遍历所有 request，看 `scheduler.is_finished(rid)` 再做清理。并发一大这个就很不优雅。

所以这次做一个很小的 mini PR：**benchmark 加 TTFT/TPOT；server 用 finished-event queue 清理请求**。

## 代码变更

### `roseinfer/engine.py`

核心思路：finished 的时刻只会发生在 `OnlineScheduler.step()` 里，所以干脆在这里把“刚刚完成的 request id”收集出来，提供一个 `pop_finished_ids()` 给外部拉取。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index c14ba5e..1d22c3e 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1393,6 +1393,7 @@ class OnlineScheduler:
         self._sessions: dict[int, InferenceSession] = {}
         self._next_request_id: int = 0
         self._round_robin_pos: int = 0
+        self._finished_ids: list[int] = []
@@ -1451,15 +1452,23 @@ class OnlineScheduler:
         last_logits = self.engine.decode_step_sessions(sessions)
         step_tokens: dict[int, int] = {}
+        just_finished: list[int] = []
         for idx, (rid, sess) in enumerate(selected_pairs):
             logits_row = last_logits[idx]
             token_id = sess.apply_batch_logits(logits_row)
             if token_id is not None:
                 step_tokens[rid] = token_id
                 if sess.finished:
+                    just_finished.append(rid)
                     sess.release_kv_blocks()
+        if just_finished:
+            self._finished_ids.extend(just_finished)
         return step_tokens
+
+    def pop_finished_ids(self) -> list[int]:
+        ids, self._finished_ids = self._finished_ids, []
+        return ids
@@ -1469,6 +1478,10 @@ class OnlineScheduler:
         session.release_kv_blocks()
         return session.decode_text()
+
+    def discard_request(self, request_id: int) -> None:
+        session = self._sessions.pop(request_id, None)
+        if session is not None:
+            session.release_kv_blocks()
```

另外顺手加了一个 `discard_request()`，用于 server 侧在 client 断开/完成后做兜底清理。

### `roseinfer/server.py`

worker loop 不再扫描 `_queues.keys()`，而是一步 decode 之后直接拿 `finished_ids`。

另外给 `stream_text()` 加了一个 `try/finally`：就算客户端中途断开，也会把 queue/detok/session 都清掉（否则会泄漏 KV blocks）。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 53dfe2b..54606e1 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -156,12 +156,21 @@ class SchedulerManager:
         return request_id
 
     def stream_text(self, request_id: int) -> Iterator[str]:
-        q = self._queues[request_id]
-        while True:
-            piece = q.get()
-            if piece is None:
-                break
-            yield piece
+        with self._lock:
+            q = self._queues.get(request_id)
+        if q is None:
+            return
+        try:
+            while True:
+                piece = q.get()
+                if piece is None:
+                    break
+                yield piece
+        finally:
+            with self._lock:
+                self._queues.pop(request_id, None)
+                self._detoks.pop(request_id, None)
+                self.scheduler.discard_request(request_id)
@@ -169,8 +178,10 @@ class SchedulerManager:
                 has_work = self.scheduler.has_unfinished()
                 if has_work:
                     step_tokens = self.scheduler.step()
+                    finished_ids = self.scheduler.pop_finished_ids()
                 else:
                     step_tokens = {}
+                    finished_ids = []
@@ -182,14 +193,11 @@ class SchedulerManager:
                 piece = detok.on_token(int(token_id))
                 if piece:
                     q.put(piece)
-            finished_ids: list[int] = []
-            with self._lock:
-                for rid in list(self._queues.keys()):
-                    if self.scheduler.is_finished(rid):
-                        finished_ids.append(rid)
             for rid in finished_ids:
-                detok = self._detoks.pop(rid, None)
-                q = self._queues.pop(rid, None)
+                with self._lock:
+                    detok = self._detoks.pop(rid, None)
+                    q = self._queues.get(rid)
+                    self.scheduler.discard_request(rid)
```

### `roseinfer/benchmark_scheduler.py`

online 模式下，我们对每个 request 记录三个时间戳：

- submit：提交请求的时刻
- first：`add_request()` 返回的时刻（因为我们会在 add_request 里同步 prefill + sample 第一个 token）
- finish：收到 finished event 的时刻

然后就能算：

- TTFT = first - submit
- Latency = finish - submit
- TPOT = (finish - first) / max(1, out_tokens - 1)

这里还顺手修了两个很经典的小坑：

1. offline summary 没有 TTFT/TPOT 字段，不能强行打印。
2. online 的 `pop_response()` 会把 session 从 `_sessions` 里 pop 掉，所以 latency 统计要在 pop 之前先把 `step_count` 读出来。

```diff
diff --git a/rosellm/roseinfer/benchmark_scheduler.py b/rosellm/roseinfer/benchmark_scheduler.py
index 22d08d3..10bf4bf 100644
--- a/rosellm/roseinfer/benchmark_scheduler.py
+++ b/rosellm/roseinfer/benchmark_scheduler.py
@@ -245,6 +245,13 @@ def summarize_runs(
     total_times = [r["elapsed"] for r in results]
     tps_decode = [r["completion_tokens"] / r["decode_elapsed"] for r in results]
     tps_total = [r["completion_tokens"] / r["elapsed"] for r in results]
+    has_latency_metrics = all(
+        k in results[0] for k in ("ttft_p50", "tpot_p50", "total_p50")
+    )
+    if has_latency_metrics:
+        ttft_p50s = [r["ttft_p50"] for r in results]
+        tpot_p50s = [r["tpot_p50"] for r in results]
+        total_p50s = [r["total_p50"] for r in results]
@@ -441,17 +464,31 @@ def benchmark_online(
 ) -> None:
-    def run_once() -> tuple[List[str], float, float]:
+    def run_once() -> tuple[
+        List[str],
+        float,
+        float,
+        List[int],
+        dict[int, float],
+        dict[int, float],
+        dict[int, float],
+        dict[int, int],
+    ]:
@@ -470,13 +512,32 @@ def benchmark_online(
         t2 = time.perf_counter()
         while scheduler.has_unfinished():
             scheduler.step()
+            now = time.perf_counter()
+            for rid in scheduler.pop_finished_ids():
+                finish_ts[rid] = now
+        for rid in request_ids:
+            if rid not in finish_ts and scheduler.is_finished(rid):
+                finish_ts[rid] = time.perf_counter()
         maybe_sync_cuda(engine)
         outputs: List[str] = []
+        for rid in request_ids:
+            sess = scheduler._sessions.get(rid)
+            if sess is not None:
+                out_tokens_by_id[rid] = sess.step_count
         for rid in request_ids:
             outputs.append(scheduler.pop_response(rid))
@@ -545,13 +606,38 @@ def benchmark_online(
     for i in range(repeat_runs):
         maybe_reset_prefix_cache(engine, args)
-        outputs, prefill_elapsed, decode_elapsed = run_once()
+        (
+            outputs,
+            prefill_elapsed,
+            decode_elapsed,
+            request_ids,
+            submit_ts,
+            first_ts,
+            finish_ts,
+            out_tokens_by_id,
+        ) = run_once()
         elapsed = prefill_elapsed + decode_elapsed
@@ -560,6 +646,9 @@ def benchmark_online(
                 elapsed=elapsed,
                 completion_tokens=completion_tokens,
                 total_tokens=total_tokens,
+                ttft_p50=ttft_p50,
+                tpot_p50=tpot_p50,
+                total_p50=total_p50,
             )
```

## 跑一下

```shell
python -m rosellm.roseinfer.benchmark_scheduler --hf-model-id gpt2 --device cpu --prompt "Hello" --num-requests 2 --max-new-tokens 4 --mode online --warmup-runs 0 --repeat-runs 1
```

输出会多三行：

```
TTFT p50/mean: 10.47/10.47 ms
TPOT p50/mean: 20.71/20.71 ms/token
Latency p50/mean: 72.59/72.59 ms
```

到这里，这个 mini PR 的目标就达到了：server 侧把“finished 检测”从 O(n) 扫描变成 O(#finished) 事件，benchmark 也终于能同时看 throughput + latency。
