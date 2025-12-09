---
classes: wide2
title: "从零实现 LLM Inference：014. Scheduler Manager"
excerpt: "实现 scheduler manager，支持 online scheduler 的接入。"
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

在实现了简单的 inference server 和初步的 openai api 接口之后，我们需要进一步接入我们的 online scheduler，之前我们的 inference server 只接了 generate 以及 stream_generate 这种旧的接口，还没有接使用了 kv-block 的 online scheduler，当接入了 online scheduler 之后，我们也就真正地能从入口请求来感受下初步的 continuous batching 能力了。



## 代码变更

主要是添加了一个 scheduler manager 来处理 server 和 online scheduler 的交互工作。

### `server.py`

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 099af58..c96e252 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -1,13 +1,16 @@
 import argparse
+import queue
+import threading
 import time
 import uuid
-from typing import List, Literal, Optional
+from typing import Dict, Iterator, List, Literal, Optional
 
 from fastapi import FastAPI
 from fastapi.responses import StreamingResponse
 from pydantic import BaseModel
 
-from .engine import InferenceEngine
+from .detokenizer import BaseDetokenizer
+from .engine import InferenceEngine, OnlineScheduler
 
 
 class GenerateRequest(BaseModel):
@@ -88,6 +91,114 @@ class ChatCompletionRequest(BaseModel):
     stream: bool = False
 
 
+class SchedulerManager:
+    def __init__(
+        self,
+        engine: InferenceEngine,
+        max_batch_size: int = 8,
+    ) -> None:
+        self.engine = engine
+        self.scheduler = OnlineScheduler(
+            engine,
+            max_batch_size=max_batch_size,
+        )
+        self._lock = threading.Lock()
+        self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
+        self._detoks: Dict[int, BaseDetokenizer] = {}
+        self._running = True
+        self._worker = threading.Thread(
+            target=self._worker_loop,
+            daemon=True,
+        )
+        self._worker.start()
+
+    def add_request(
+        self,
+        prompt: str,
+        max_new_tokens: int = 64,
+        temperature: float = 1.0,
+        top_k: int = 0,
+        top_p: float = 1.0,
+        stop_on_eos: bool = True,
+        do_sample: bool = False,
+    ) -> int:
+        token_ids = self.engine.tokenizer.encode(
+            prompt,
+            add_special_tokens=False,
+        )
+        if not token_ids:
+            token_ids = [self.engine.eos_token_id]
+        detok = self.engine._make_detok()
+        detok.start_prompt(token_ids)
+        with self._lock:
+            request_id = self.scheduler.add_request(
+                prompt=prompt,
+                max_new_tokens=max_new_tokens,
+                temperature=temperature,
+                top_k=top_k,
+                top_p=top_p,
+                stop_on_eos=stop_on_eos,
+                do_sample=do_sample,
+            )
+            q: "queue.Queue[Optional[str]]" = queue.Queue()
+            self._queues[request_id] = q
+            self._detoks[request_id] = detok
+            session = self.scheduler._sessions[request_id]
+            for tid in session.generated_ids:
+                piece = detok.on_token(int(tid))
+                if piece:
+                    q.put(piece)
+            if session.finished:
+                tail = detok.flush()
+                if tail:
+                    q.put(tail)
+                q.put(None)
+        return request_id
+
+    def stream_text(self, request_id: int) -> Iterator[str]:
+        q = self._queues[request_id]
+        while True:
+            piece = q.get()
+            if piece is None:
+                break
+            yield piece
+
+    def _worker_loop(self) -> None:
+        while self._running:
+            with self._lock:
+                has_work = self.scheduler.has_unfinished()
+                if has_work:
+                    step_tokens = self.scheduler.step()
+                else:
+                    step_tokens = {}
+            if not has_work:
+                time.sleep(0.005)
+                continue
+            for rid, token_id in step_tokens.items():
+                detok = self._detoks.get(rid)
+                q = self._queues.get(rid)
+                if detok is None or q is None:
+                    continue
+                piece = detok.on_token(int(token_id))
+                if piece:
+                    q.put(piece)
+            finished_ids: list[int] = []
+            with self._lock:
+                for rid in list(self._queues.keys()):
+                    if self.scheduler.is_finished(rid):
+                        finished_ids.append(rid)
+            for rid in finished_ids:
+                detok = self._detoks.pop(rid, None)
+                q = self._queues.pop(rid, None)
+                if q is None:
+                    continue
+                if detok is not None:
+                    tail = detok.flush()
+                    if tail:
+                        q.push(tail)
+                q.put(None)
+
+
 def format_messages_as_prompt(messages: List[ChatMessage]) -> str:
     lines: list[str] = []
     for m in messages:
@@ -120,6 +231,7 @@ def estimate_usage(
 
 def create_app(engine: InferenceEngine) -> FastAPI:
     app = FastAPI(title="roseinfer", version="0.1.0")
+    sched_manager = SchedulerManager(engine, max_batch_size=8)
 
     @app.get("/health")
     def health() -> dict[str, str]:
@@ -130,18 +242,18 @@ def create_app(engine: InferenceEngine) -> FastAPI:
         body: GenerateRequest,
     ) -> GenerateResponse | StreamingResponse:
         if body.stream:
+            request_id = sched_manager.add_request(
+                prompt=body.prompt,
+                max_new_tokens=body.max_new_tokens,
+                temperature=body.temperature,
+                top_k=body.top_k,
+                top_p=body.top_p,
+                stop_on_eos=body.stop_on_eos,
+                do_sample=body.do_sample,
+            )
 
-            def token_stream() -> bytes:
-                chunks = engine.stream_generate(
-                    prompt=body.prompt,
-                    max_new_tokens=body.max_new_tokens,
-                    temperature=body.temperature,
-                    top_k=body.top_k,
-                    top_p=body.top_p,
-                    stop_on_eos=body.stop_on_eos,
-                    do_sample=body.do_sample,
-                )
-                for piece in chunks:
+            def token_stream() -> Iterator[bytes]:
+                for piece in sched_manager.stream_text(request_id):
                     yield piece.encode("utf-8")
 
             return StreamingResponse(
@@ -168,6 +280,15 @@ def create_app(engine: InferenceEngine) -> FastAPI:
         completion_id = f"chatcmpl-{uuid.uuid4().hex}"
         model_name = body.model or "roseinfer"
         if body.stream:
+            request_id = sched_manager.add_request(
+                prompt=prompt,
+                max_new_tokens=body.max_tokens,
+                temperature=body.temperature,
+                top_k=body.top_k,
+                top_p=body.top_p,
+                stop_on_eos=True,
+                do_sample=True,
+            )
 
             def event_stream():
                 first_chunk = ChatCompletionChunk(
@@ -187,15 +308,7 @@ def create_app(engine: InferenceEngine) -> FastAPI:
                     ],
                 )
                 yield f"data: {first_chunk.model_dump_json()}\n\n".encode("utf-8")
-                for piece in engine.stream_generate(
-                    prompt=prompt,
-                    max_new_tokens=body.max_tokens,
-                    temperature=body.temperature,
-                    top_k=body.top_k,
-                    top_p=body.top_p,
-                    stop_on_eos=True,
-                    do_sample=True,
-                ):
+                for piece in sched_manager.stream_text(request_id):
                     if not piece:
                         continue
                     chunk = ChatCompletionChunk(

```



## 运行

再类似之前一样运行：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ python -m rosellm.roseinfer.server --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt --tokenizer-name gpt2 --device cuda --port 8080
INFO:     Started server process [1625699]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     127.0.0.1:54660 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

进行请求：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ curl -X POST "http://127.0.0.1:8080/v1/chat/completions"   -H "Content-Type: application/json"   -d '{
    "model": "roseinfer-gpt2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": true
  }'
data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" "},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"A"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"."},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" Why"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" the"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" client"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" not"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" to"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" attend"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" a"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" high"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" school"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"?"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"A"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"."},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" What"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" the"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" client"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"?"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"A"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"."},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" What"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" the"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" client"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"?"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-147a5660b7544006ae0d91499894913d","object":"chat.completion.chunk","created":1765281448,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":null},"finish_reason":"stop"}]}

data: [DONE]

(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ 
```

