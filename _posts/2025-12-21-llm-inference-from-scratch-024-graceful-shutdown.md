---
classes: wide2
title: "从零实现 LLM Inference：024. Graceful Shutdown"
excerpt: "补齐 SchedulerManager 的 close，优雅停掉 worker 并结束 streaming。"
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

上一版我们把 server worker 从 `time.sleep(0.005)` 改成了 `Event.wait()`，idle 时不再轮询。

但只要你把 wait 引进来，就必须补上一个闭环：**怎么停**。否则 shutdown 的时候，worker 有概率卡在 wait 里，或者 streaming 请求一直挂着不结束。

所以这次做一个很小的 mini PR：给 `SchedulerManager` 加一个 `close()`，并在 FastAPI shutdown 时自动调用它。

## 代码变更

### `roseinfer/server.py`

核心点：

1. `close()` 里把 `_running=False`，并 `set()` 唤醒 worker。
2. 给所有还活着的 request queue `put(None)`，让 `stream_text()` 尽快结束。
3. shutdown hook：`app.add_event_handler("shutdown", sched_manager.close)`。
4. worker loop 里在锁内先检查 `_running`，避免 stop 时把 wakeup 信号误 clear 掉。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index b7544ee..bc8028c 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -113,6 +113,23 @@ class SchedulerManager:
         )
         self._worker.start()
 
+    def close(self) -> None:
+        worker = self._worker
+        with self._lock:
+            if not self._running:
+                return
+            self._running = False
+            self._wakeup.set()
+            request_ids = list(self._queues.keys())
+            for rid in request_ids:
+                q = self._queues.get(rid)
+                if q is not None:
+                    q.put(None)
+                self.scheduler.discard_request(rid)
+            self._queues.clear()
+            self._detoks.clear()
+        worker.join(timeout=1.0)
@@ -176,8 +193,10 @@ class SchedulerManager:
                 self.scheduler.discard_request(request_id)
 
     def _worker_loop(self) -> None:
-        while self._running:
+        while True:
             with self._lock:
+                if not self._running:
+                    break
                 has_work = self.scheduler.has_unfinished()
                 if has_work:
                     step_tokens = self.scheduler.step()
@@ -244,6 +263,7 @@ def estimate_usage(
 def create_app(engine: InferenceEngine) -> FastAPI:
     app = FastAPI(title="roseinfer", version="0.1.0")
     sched_manager = SchedulerManager(engine, max_batch_size=8)
+    app.add_event_handler("shutdown", sched_manager.close)
```

## 运行

启动 server：

```shell
python -m rosellm.roseinfer.server \
  --checkpoint-path rosellm/rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt \
  --tokenizer-name gpt2 \
  --device cpu \
  --host 127.0.0.1 \
  --port 8090
```

起一个 streaming 请求：

```shell
curl -N -X POST http://127.0.0.1:8090/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":128,"stream":true}'
```

输出（节选）：

```text
, you’re going to be a little bit more than a little bit more than a little bit more.
```

此时在 server 侧 `Ctrl+C`，可以看到 uvicorn 会正常 shutdown，streaming 也会随之结束，不会一直挂着。
