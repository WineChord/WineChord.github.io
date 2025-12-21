---
classes: wide2
title: "从零实现 LLM Inference：022. Online Scheduler Active Queue"
excerpt: "OnlineScheduler 用 deque 维护活跃队列，去掉每 step 的全量扫描。"
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

之前 OnlineScheduler 的 `step()` 每次都会从 `_sessions` 全量扫一遍，把所有未 finished 的 session 重新收集出来，再做 round-robin 选取。

这个实现的好处是写起来直观，但问题也很明显：并发一大，**每 step 都有一段纯 CPU/Python 的 O(n) 开销**，而且 server worker 还需要持锁做这些事，容易把 decode 的节奏拖散。

所以这次做一个很小的 mini PR：**维护一个活跃队列（active queue）**，每步只处理 batch_size 个 request，把复杂度从 “每 step 扫所有请求” 降到 “每 step 只碰 batch_size 个请求（外加少量惰性清理）”。

## 代码变更

### `roseinfer/engine.py`

核心就是引入 `deque`：

1. `add_request()`：新请求如果没 finished，就把 rid append 到 `_active_rids`。
2. `has_unfinished()`：只看 `_active_rids`，并且顺手把队首已经结束/被丢弃的 rid pop 掉（惰性清理）。
3. `step()`：从 `_active_rids` 里 `popleft()` 出最多 `max_batch_size` 个活跃 rid，跑一次 batched decode；没结束的再 append 回队尾，天然形成 round-robin。
4. 顺手把 `is_finished()` 做得更健壮：如果 session 已经被 pop/discard，就按 finished 处理。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 1d22c3e..7cf9382 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1,4 +1,4 @@
-from collections import OrderedDict
+from collections import OrderedDict, deque
 from typing import Iterator, NamedTuple, Optional
 
 import torch
@@ -1389,10 +1389,10 @@ class OnlineScheduler:
         self.max_batch_size = max_batch_size
         self.use_prefix_cache = use_prefix_cache
         self._sessions: dict[int, InferenceSession] = {}
         self._next_request_id: int = 0
-        self._round_robin_pos: int = 0
+        self._active_rids: deque[int] = deque()
         self._finished_ids: list[int] = []
@@ -1425,6 +1425,8 @@ class OnlineScheduler:
         request_id = self._next_request_id
         self._next_request_id += 1
         self._sessions[request_id] = session
+        if not session.finished:
+            self._active_rids.append(request_id)
         return request_id
@@ -1430,7 +1432,15 @@ class OnlineScheduler:
-        return any(not sess.finished for sess in self._sessions.values())
+        while self._active_rids:
+            rid = self._active_rids[0]
+            sess = self._sessions.get(rid)
+            if sess is None or sess.finished:
+                self._active_rids.popleft()
+                continue
+            return True
+        return False
@@ -1433,7 +1443,10 @@ class OnlineScheduler:
-        session = self._sessions.get(request_id, None)
+        session = self._sessions.get(request_id)
+        if session is None:
+            return True
         return session.finished
@@ -1437,7 +1447,18 @@ class OnlineScheduler:
-        active_pairs: list[tuple[int, InferenceSession]] = [
-            (rid, sess) for rid, sess in self._sessions.items() if not sess.finished
-        ]
-        if not active_pairs:
+        if not self._active_rids:
             return {}
-        num_active = len(active_pairs)
-        batch_size = min(self.max_batch_size, num_active)
-        start = self._round_robin_pos % num_active
-        selected_pairs: list[tuple[int, InferenceSession]] = []
-        for i in range(batch_size):
-            idx = (start + i) % num_active
-            selected_pairs.append(active_pairs[idx])
-        self._round_robin_pos = (start + batch_size) % num_active
+        selected_pairs: list[tuple[int, InferenceSession]] = []
+        max_examine = len(self._active_rids)
+        while (
+            len(selected_pairs) < self.max_batch_size
+            and self._active_rids
+            and max_examine > 0
+        ):
+            max_examine -= 1
+            rid = self._active_rids.popleft()
+            sess = self._sessions.get(rid)
+            if sess is None or sess.finished:
+                continue
+            selected_pairs.append((rid, sess))
+        if not selected_pairs:
+            return {}
@@ -1460,6 +1481,8 @@ class OnlineScheduler:
                 if sess.finished:
                     just_finished.append(rid)
                     sess.release_kv_blocks()
+                else:
+                    self._active_rids.append(rid)
```

这里还有一个很关键的取舍：`discard_request()` 不需要从 deque 里 O(n) 删除 rid，而是让它“留在队列里”，后续 `has_unfinished/step` 再遇到时自然跳过（惰性删除）。这也是很多调度系统里常用的做法：**保持 fast path 简单，清理做摊还**。

## 运行

```shell
python -m rosellm.roseinfer.benchmark_scheduler --hf-model-id gpt2 --device cpu --prompt "Hello" --num-requests 8 --max-new-tokens 8 --mode online --warmup-runs 0 --repeat-runs 1
```

输出：

```text
=== online summary ===
Warmup runs: 0
Measured runs: 1
Decode time p50/mean: 0.172318/0.172318 s
Total time p50/mean: 0.194708/0.194708 s
Throughput(completion,decode) p50/mean: 371.41/371.41 tokens/s
Throughput(completion,total) p50/mean: 328.70/328.70 tokens/s
TTFT p50/mean: 0.25/0.25 ms
TPOT p50/mean: 24.69/24.69 ms/token
Latency p50/mean: 173.08/173.08 ms
```
