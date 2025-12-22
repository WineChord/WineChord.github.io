---
classes: wide2
title: "从零实现 LLM Inference：027. Pending Queue Admission"
excerpt: "把 streaming 的 add_request 从 prefill 中解耦：快速入队，由 worker 统一做 prefill + decode。"
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

这一版解决一个很现实的问题：**streaming 的 add_request 太慢**。

之前 `SchedulerManager.add_request()` 会在锁内直接跑 `OnlineScheduler.add_request()`，而后者会做 prefill（一次 model forward）。这意味着 burst 提交时会被硬串行化：客户端在“入队”阶段就被卡住。

所以这次 mini PR 目标非常明确：**add_request 只做轻量工作（tokenize + 建 queue + 入 pending queue），prefill/step 全部挪到 worker 线程里**。

## Benchmark（HF GPT-2）

命令：

```shell
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --unique-prompts \
  --num-requests 32 \
  --max-new-tokens 8 \
  --no-stop-on-eos \
  --no-prefix-cache
```

### Before

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 1.625954 s
add_request latency p50/p95/p99: 47.04/74.25/108.53 ms
TTFT p50/p95/p99: 71.87/121.18/141.16 ms
Latency p50/p95/p99: 401.74/455.18/467.88 ms
Throughput (completion,total): 142.58 tokens/s
```

### After

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.085824 s
add_request latency p50/p95/p99: 0.05/0.16/54.87 ms
TTFT p50/p95/p99: 403.11/750.30/775.59 ms
Latency p50/p95/p99: 1393.64/1463.76/1463.94 ms
Throughput (completion,total): 165.28 tokens/s
```

结论很清楚：

- **Submit wall：`1.626s -> 0.086s`（~19x）**，burst 提交不再被 prefill 串行卡死。
- **add_request p50：`47.04ms -> 0.05ms`（~940x）**，入队路径基本只剩下 tokenization + 少量 bookkeeping。

TTFT/Latency 这次看起来会变大，是因为 “Before” 的提交本身就是慢的（每个请求的 submit_start 被拉开了 1.6s），很多排队时间被藏在了提交阶段；“After” 则是真正把 32 个请求快速入队后，排队时间才会自然地体现在 TTFT/Latency 上。下一步如果要继续压 TTFT，就该去做 **batched prefill** / **prefill-decode pipeline** 了。

## 代码变更

### `roseinfer/server.py`

核心改动：

1. 新增 `_PendingRequest` + `_pending` 队列。
2. `SchedulerManager.add_request()` 只负责分配 `request_id`、建立 queue/detok，然后把请求丢进 pending。
3. worker loop 里 drain pending，再调用 `OnlineScheduler.add_request()` 做 prefill；之后继续跑 decode step 并把 token 推到对应 queue。
4. worker 异常时，`put(None)` 结束所有 streaming，避免客户端永远卡住。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 2cabcfc..81f29d8 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -2,7 +2,9 @@ import argparse
 import queue
 import threading
 import time
+import traceback
 import uuid
+from dataclasses import dataclass
 from typing import Dict, Iterator, List, Literal, Optional
@@ -91,6 +93,19 @@ class ChatCompletionRequest(BaseModel):
     stream: bool = False
 
 
+@dataclass(frozen=True)
+class _PendingRequest:
+    request_id: int
+    prompt: str
+    prompt_token_ids: list[int]
+    max_new_tokens: int
+    temperature: float
+    top_k: int
+    top_p: float
+    stop_on_eos: bool
+    do_sample: bool
+
+
 class SchedulerManager:
     def __init__(
@@ -106,6 +121,8 @@ class SchedulerManager:
         self._wakeup = threading.Event()
         self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
         self._detoks: Dict[int, BaseDetokenizer] = {}
+        self._pending: "queue.Queue[_PendingRequest]" = queue.Queue()
+        self._next_request_id: int = 0
         self._running = True
         self._worker = threading.Thread(
@@ -115,6 +132,7 @@ class SchedulerManager:
 
     def close(self) -> None:
         worker = self._worker
+        request_ids: list[int] = []
         with self._lock:
             if not self._running:
                 return
@@ -125,10 +143,13 @@ class SchedulerManager:
                 q = self._queues.get(rid)
                 if q is not None:
                     q.put(None)
-                self.scheduler.discard_request(rid)
             self._queues.clear()
             self._detoks.clear()
         worker.join(timeout=1.0)
+        if worker.is_alive():
+            return
+        for rid in request_ids:
+            self.scheduler.discard_request(rid)
 
     def add_request(
         self,
@@ -149,30 +170,27 @@ class SchedulerManager:
         detok = self.engine._make_detok()
         detok.start_prompt(token_ids)
         with self._lock:
-            request_id = self.scheduler.add_request(
+            if not self._running:
+                raise RuntimeError("SchedulerManager is closed")
+            request_id = self._next_request_id
+            self._next_request_id += 1
+            q: "queue.Queue[Optional[str]]" = queue.Queue()
+            self._queues[request_id] = q
+            self._detoks[request_id] = detok
+        self._pending.put(
+            _PendingRequest(
+                request_id=request_id,
                 prompt=prompt,
+                prompt_token_ids=list(token_ids),
                 max_new_tokens=max_new_tokens,
                 temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 stop_on_eos=stop_on_eos,
                 do_sample=do_sample,
-                prompt_token_ids=token_ids,
             )
-            q: "queue.Queue[Optional[str]]" = queue.Queue()
-            self._queues[request_id] = q
-            self._detoks[request_id] = detok
-            for tid in self.scheduler.get_generated_ids(request_id):
-                piece = detok.on_token(int(tid))
-                if piece:
-                    q.put(piece)
-            if self.scheduler.is_finished(request_id):
-                tail = detok.flush()
-                if tail:
-                    q.put(tail)
-                q.put(None)
-            else:
-                self._wakeup.set()
+        )
+        self._wakeup.set()
         return request_id
 
     def stream_text(self, request_id: int) -> Iterator[str]:
@@ -190,43 +208,97 @@ class SchedulerManager:
             with self._lock:
                 self._queues.pop(request_id, None)
                 self._detoks.pop(request_id, None)
-                self.scheduler.discard_request(request_id)
 
     def _worker_loop(self) -> None:
-        while True:
-            with self._lock:
-                if not self._running:
-                    break
-                has_work = self.scheduler.has_unfinished()
-                if has_work:
+        try:
+            while True:
+                with self._lock:
+                    if not self._running:
+                        break
+                    max_new = self.scheduler.max_batch_size
+                pending: list[_PendingRequest] = []
+                for _ in range(max_new):
+                    try:
+                        pending.append(self._pending.get_nowait())
+                    except queue.Empty:
+                        break
+                for req in pending:
+                    with self._lock:
+                        if not self._running:
+                            break
+                        q = self._queues.get(req.request_id)
+                        detok = self._detoks.get(req.request_id)
+                    if q is None or detok is None:
+                        continue
+                    rid = self.scheduler.add_request(
+                        prompt=req.prompt,
+                        max_new_tokens=req.max_new_tokens,
+                        temperature=req.temperature,
+                        top_k=req.top_k,
+                        top_p=req.top_p,
+                        stop_on_eos=req.stop_on_eos,
+                        do_sample=req.do_sample,
+                        prompt_token_ids=req.prompt_token_ids,
+                        request_id=req.request_id,
+                    )
+                    with self._lock:
+                        q = self._queues.get(rid)
+                        detok = self._detoks.get(rid)
+                    if q is None or detok is None:
+                        self.scheduler.discard_request(rid)
+                        continue
+                    for tid in self.scheduler.get_generated_ids(rid):
+                        piece = detok.on_token(int(tid))
+                        if piece:
+                            q.put(piece)
+                    if self.scheduler.is_finished(rid):
+                        tail = detok.flush()
+                        if tail:
+                            q.put(tail)
+                        q.put(None)
+                        self.scheduler.discard_request(rid)
+
+                if self.scheduler.has_unfinished():
                     step_tokens = self.scheduler.step()
                     finished_ids = self.scheduler.pop_finished_ids()
                 else:
                     step_tokens = {}
                     finished_ids = []
-                    self._wakeup.clear()
-            if not has_work:
-                self._wakeup.wait()
-                continue
-            for rid, token_id in step_tokens.items():
-                detok = self._detoks.get(rid)
-                q = self._queues.get(rid)
-                if detok is None or q is None:
-                    continue
-                piece = detok.on_token(int(token_id))
-                if piece:
-                    q.put(piece)
-            for rid in finished_ids:
-                with self._lock:
-                    detok = self._detoks.pop(rid, None)
-                    q = self._queues.get(rid)
+
+                for rid, token_id in step_tokens.items():
+                    with self._lock:
+                        q = self._queues.get(rid)
+                        detok = self._detoks.get(rid)
+                    if q is None or detok is None:
+                        self.scheduler.discard_request(rid)
+                        continue
+                    piece = detok.on_token(int(token_id))
+                    if piece:
+                        q.put(piece)
+
+                for rid in finished_ids:
+                    with self._lock:
+                        q = self._queues.get(rid)
+                        detok = self._detoks.get(rid)
                     self.scheduler.discard_request(rid)
-                if q is None:
-                    continue
-                if detok is not None:
-                    tail = detok.flush()
-                    if tail:
-                        q.put(tail)
+                    if q is None:
+                        continue
+                    if detok is not None:
+                        tail = detok.flush()
+                        if tail:
+                            q.put(tail)
+                    q.put(None)
+
+                if not pending and not self.scheduler.has_unfinished():
+                    self._wakeup.wait()
+                    self._wakeup.clear()
+        except Exception:
+            traceback.print_exc()
+            with self._lock:
+                queues = list(self._queues.values())
+                self._running = False
+                self._wakeup.set()
+            for q in queues:
                 q.put(None)
```

### `roseinfer/engine.py`

为了让 server 在 “入队时” 就能分配 `request_id` 并把 queue/detok 建好，`OnlineScheduler.add_request()` 新增可选参数 `request_id`，允许外部指定 rid（同时做冲突检查）。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 8aaae0b..34e2370 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ class OnlineScheduler:
     def add_request(
         self,
@@
         prompt_token_ids: Optional[list[int]] = None,
+        request_id: Optional[int] = None,
     ) -> int:
@@
-        request_id = self._next_request_id
-        self._next_request_id += 1
-        self._sessions[request_id] = session
+        if request_id is None:
+            rid = self._next_request_id
+            self._next_request_id += 1
+        else:
+            rid = int(request_id)
+            if rid in self._sessions:
+                raise ValueError(f"request_id {rid} already exists")
+            if rid >= self._next_request_id:
+                self._next_request_id = rid + 1
+        self._sessions[rid] = session
@@
-            self._active_rids.append(request_id)
-        return request_id
+            self._active_rids.append(rid)
+        return rid
```

### `roseinfer/benchmark_streaming.py`

新增一个专门看 streaming admission/TTFT 的小 benchmark，用来把 “入队阶段被卡住” 这种问题量化出来。

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
new file mode 100644
--- /dev/null
+++ b/rosellm/roseinfer/benchmark_streaming.py
index 0000000..0d3570e
@@ -0,0 +1,286 @@
+import argparse
+import math
+import statistics
+import threading
+import time
+from dataclasses import dataclass
+from typing import List
+
+import torch
+
+from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained
+
+from .engine import InferenceEngine
+from .server import SchedulerManager
+
+        add_lats = [r.submit_end - r.submit_start for r in results]
+        ttfts = [r.first_token_ts - r.submit_start for r in results]
+        totals = [r.finish_ts - r.submit_start for r in results]
+        completion_tokens = [r.completion_tokens for r in results]
+        sum_tokens = sum(completion_tokens)
+
+        print("=== streaming benchmark ===")
+        print(f"Model: {args.hf_model_id}")
+        print(f"Device: {args.device}")
+        print(f"Requests: {len(results)}")
+        print(f"Prompt tokens (total): {sum(prompt_lens)}")
+        print(f"Completion tokens (total): {sum_tokens}")
+        print(f"Submit wall: {submit_wall:.6f} s")
+        print(
+            f"add_request latency p50/p95/p99: "
+            f"{statistics.median(add_lats)*1e3:.2f}/"
+            f"{percentile(add_lats, 95)*1e3:.2f}/"
+            f"{percentile(add_lats, 99)*1e3:.2f} ms"
+        )
+        print(
+            f"TTFT p50/p95/p99: "
+            f"{statistics.median(ttfts)*1e3:.2f}/"
+            f"{percentile(ttfts, 95)*1e3:.2f}/"
+            f"{percentile(ttfts, 99)*1e3:.2f} ms"
+        )
+        print(
+            f"Latency p50/p95/p99: "
+            f"{statistics.median(totals)*1e3:.2f}/"
+            f"{percentile(totals, 95)*1e3:.2f}/"
+            f"{percentile(totals, 99)*1e3:.2f} ms"
+        )
+        print(f"Throughput (completion,total): {sum_tokens / wall:.2f} tokens/s")
```

### `tests/test_server_streaming.py`

补一个最小单测：起一个 streaming 请求，确保能吐出 token，并且最终结束。

```diff
diff --git a/tests/test_server_streaming.py b/tests/test_server_streaming.py
new file mode 100644
--- /dev/null
+++ b/tests/test_server_streaming.py
index 0000000..fa805e3
@@ -0,0 +1,65 @@
+import torch
+
+from rosellm.roseinfer.engine import InferenceEngine
+from rosellm.roseinfer.server import SchedulerManager
+from rosellm.rosetrainer.config import GPTConfig
+from rosellm.rosetrainer.model import GPTModel
+
+
+class _DummyTokenizer:
+    def __init__(self, vocab_size: int = 128) -> None:
+        self.vocab_size = int(vocab_size)
+        self.eos_token_id = 0
+        self.pad_token_id = 0
+
+    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
+        del add_special_tokens
+        if not text:
+            return [self.eos_token_id]
+        out: list[int] = []
+        for b in text.encode("utf-8"):
+            out.append(int(b % (self.vocab_size - 1)) + 1)
+        return out
+
+    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
+        del skip_special_tokens
+        return " ".join(str(i) for i in ids)
+
+
+def test_server_streaming_emits_tokens() -> None:
+    torch.manual_seed(0)
+    cfg = GPTConfig(
+        vocab_size=128,
+        max_position_embeddings=32,
+        n_layers=2,
+        n_heads=2,
+        d_model=32,
+        d_ff=64,
+        dropout=0.0,
+    )
+    tok = _DummyTokenizer(vocab_size=128)
+    model = GPTModel(cfg)
+    engine = InferenceEngine(
+        model=model,
+        config=cfg,
+        tokenizer=tok,
+        tokenizer_name="dummy",
+        device="cpu",
+        use_amp=False,
+        kv_cache_max_concurrency=4,
+        prefix_cache_max_entries=0,
+    )
+
+    mgr = SchedulerManager(engine, max_batch_size=2)
+    try:
+        mgr.scheduler.use_prefix_cache = False
+        rid = mgr.add_request(
+            "hello",
+            max_new_tokens=2,
+            stop_on_eos=False,
+            do_sample=False,
+        )
+        pieces = list(mgr.stream_text(rid))
+        assert "".join(pieces) != ""
+    finally:
+        mgr.close()
```

## 运行

```shell
pytest -q
```

输出：

```text
......                                                                   [100%]
6 passed, 1 warning in 1.59s
```
