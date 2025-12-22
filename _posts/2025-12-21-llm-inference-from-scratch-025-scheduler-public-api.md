---
classes: wide2
title: "从零实现 LLM Inference：025. Scheduler Public API"
excerpt: "把 server/benchmark 对 OnlineScheduler._sessions 的直接访问收口。"
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

这次是一个非常小的清理：把 `server.py` / `benchmark_scheduler.py` 里对 `OnlineScheduler._sessions` 的直接访问去掉。

理由很简单：`_sessions` 本质是 scheduler 的内部实现细节，外层一旦“摸进去”，后面 scheduler 想换数据结构、做并发控制、做 session 生命周期管理，都会被上层耦住。

所以这次只做一件事：加两个只读 accessor，把需要的最小信息暴露出来。

## 代码变更

### `roseinfer/engine.py`

给 `OnlineScheduler` 增加两个方法：

- `get_generated_ids(request_id)`：拿到当前已生成 token 的 snapshot（返回 copy）
- `get_step_count(request_id)`：拿到当前已生成 token 数（prefill 的第 1 个 token 也算在内）

### `roseinfer/server.py`

`SchedulerManager.add_request()` 里不再读取 `scheduler._sessions[request_id]`，改为：

1. `get_generated_ids()` 把 prefill 已经吐出来的 token 先塞进 queue；
2. `is_finished()` 判断是否已经结束，决定要不要 `flush()` + `put(None)`。

### `roseinfer/benchmark_scheduler.py`

online benchmark 里统计 `out_tokens_by_id` 时，不再去读 `scheduler._sessions`，改为 `get_step_count()`。

### `tests/`

补了一个很小的单测：验证 `get_generated_ids()` 返回的是 copy，不会被外层误改。

```diff
diff --git a/rosellm/roseinfer/benchmark_scheduler.py b/rosellm/roseinfer/benchmark_scheduler.py
index 10bf4bf..ce47fff 100644
--- a/rosellm/roseinfer/benchmark_scheduler.py
+++ b/rosellm/roseinfer/benchmark_scheduler.py
@@ -521,9 +521,7 @@ def benchmark_online(
         maybe_sync_cuda(engine)
         outputs: List[str] = []
         for rid in request_ids:
-            sess = scheduler._sessions.get(rid)
-            if sess is not None:
-                out_tokens_by_id[rid] = sess.step_count
+            out_tokens_by_id[rid] = scheduler.get_step_count(rid)
         for rid in request_ids:
             outputs.append(scheduler.pop_response(rid))
         t3 = time.perf_counter()
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index b986fa1..3cbd12d 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1445,6 +1445,18 @@ class OnlineScheduler:
         if session is None:
             return True
         return session.finished
+
+    def get_generated_ids(self, request_id: int) -> list[int]:
+        session = self._sessions.get(request_id)
+        if session is None:
+            return []
+        return list(session.generated_ids)
+
+    def get_step_count(self, request_id: int) -> int:
+        session = self._sessions.get(request_id)
+        if session is None:
+            return 0
+        return int(session.step_count)
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index bc8028c..7b6a71f 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -161,12 +161,11 @@ class SchedulerManager:
             q: "queue.Queue[Optional[str]]" = queue.Queue()
             self._queues[request_id] = q
             self._detoks[request_id] = detok
-            session = self.scheduler._sessions[request_id]
-            for tid in session.generated_ids:
+            for tid in self.scheduler.get_generated_ids(request_id):
                 piece = detok.on_token(int(tid))
                 if piece:
                     q.put(piece)
-            if session.finished:
+            if self.scheduler.is_finished(request_id):
                 tail = detok.flush()
                 if tail:
                     q.put(tail)
```

## 运行

```shell
python -m rosellm.roseinfer.benchmark_scheduler --hf-model-id gpt2 --device cpu --prompt "Hello" --num-requests 8 --max-new-tokens 8 --mode online --warmup-runs 0 --repeat-runs 1
```

```text
=== online summary ===
Warmup runs: 0
Measured runs: 1
Decode time p50/mean: 0.171313/0.171313 s
Total time p50/mean: 0.193439/0.193439 s
Throughput(completion,decode) p50/mean: 373.59/373.59 tokens/s
Throughput(completion,total) p50/mean: 330.85/330.85 tokens/s
TTFT p50/mean: 0.26/0.26 ms
TPOT p50/mean: 24.55/24.55 ms/token
Latency p50/mean: 172.09/172.09 ms
```
