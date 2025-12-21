---
classes: wide2
title: "从零实现 LLM Inference：023. Worker Wakeup Event"
excerpt: "server worker 用 threading.Event 驱动唤醒，去掉 idle 轮询 sleep。"
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

在 `SchedulerManager._worker_loop()` 里，之前我用了一个很粗暴的 idle 方案：没活就 `time.sleep(0.005)`，然后继续轮询。

这个写法简单，但有两个问题：

1. 新请求进来后，worker 最多会晚 5ms 才开始 decode（对 streaming 的首包很不友好）。
2. 空闲时会不断 wake up（哪怕工作量不大，也是不必要的开销）。

所以这次做一个很小的 mini PR：**用 `threading.Event` 把 worker 改成事件驱动唤醒**。

## 代码变更

### `roseinfer/server.py`

核心点只有三个：

1. `SchedulerManager` 新增 `_wakeup = threading.Event()`。
2. `add_request()`：只有当 request 需要继续 decode（session 没 finished）时 `set()`。
3. worker idle 时 `clear()` 并 `wait()`，不再 sleep 轮询。

这里 `clear()` 必须放在锁里做，避免丢唤醒信号。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 54606e1..b7544ee 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -103,6 +103,7 @@ class SchedulerManager:
             max_batch_size=max_batch_size,
         )
         self._lock = threading.Lock()
+        self._wakeup = threading.Event()
         self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
         self._detoks: Dict[int, BaseDetokenizer] = {}
         self._running = True
@@ -153,6 +154,8 @@ class SchedulerManager:
                 if tail:
                     q.put(tail)
                 q.put(None)
+            else:
+                self._wakeup.set()
         return request_id
@@ -182,8 +185,9 @@ class SchedulerManager:
                 else:
                     step_tokens = {}
                     finished_ids = []
+                    self._wakeup.clear()
             if not has_work:
-                time.sleep(0.005)
+                self._wakeup.wait()
                 continue
             for rid, token_id in step_tokens.items():
                 detok = self._detoks.get(rid)
```

## 运行

启动 server：

```shell
python -m rosellm.roseinfer.server \
  --checkpoint-path rosellm/rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt \
  --tokenizer-name gpt2 \
  --device cpu \
  --host 127.0.0.1 \
  --port 8089
```

输出（节选）：

```text
INFO:     Started server process [156286]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8089 (Press CTRL+C to quit)
```

发送一个 streaming 请求：

```shell
curl -N -X POST http://127.0.0.1:8089/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":8,"stream":true}'
```

输出（节选）：

```text
, you're going to be
```
