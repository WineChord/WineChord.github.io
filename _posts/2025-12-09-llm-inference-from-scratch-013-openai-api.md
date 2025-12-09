---
classes: wide2
title: "从零实现 LLM Inference：013. Simple OpenAI API"
excerpt: "支持简单的 openai api，实现 chat completion。"
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

上一个 PR 我们实现了简单的 inference server，基于 FastAPI 和 uvicorn，这个 PR 我们支持一个非常简单的 openai api server。

## 代码变更

### `server.py`

主要就是注册一个新的路由，然后处理请求，返回响应：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 84206c9..099af58 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -1,4 +1,7 @@
 import argparse
+import time
+import uuid
+from typing import List, Literal, Optional
 
 from fastapi import FastAPI
 from fastapi.responses import StreamingResponse
@@ -22,6 +25,99 @@ class GenerateResponse(BaseModel):
     text: str
 
 
+class ChatMessage(BaseModel):
+    role: Literal["system", "user", "assistant"]
+    content: str
+
+
+class UsageInfo(BaseModel):
+    prompt_tokens: int
+    completion_tokens: int
+    total_tokens: int
+
+
+class ChatCompletionChoice(BaseModel):
+    index: int
+    message: ChatMessage
+    finish_reason: Optional[str] = None
+
+
+class ChatCompletionResponse(BaseModel):
+    id: str
+    object: Literal["chat.completion"]
+    created: int
+    model: str
+    choices: List[ChatCompletionChoice]
+    usage: Optional[UsageInfo] = None
+
+
+class ChatCompletionChunkDelta(BaseModel):
+    role: Optional[str] = None
+    content: Optional[str] = None
+
+
+class ChatCompletionChunkChoice(BaseModel):
+    index: int
+    delta: ChatCompletionChunkDelta
+    finish_reason: Optional[str] = None
+
+
+class ChatCompletionChunk(BaseModel):
+    id: str
+    object: Literal["chat.completion.chunk"]
+    created: int
+    model: str
+    choices: List[ChatCompletionChunkChoice]
+
+
+class ChatCompletionChunkResponse(BaseModel):
+    id: str
+    object: Literal["chat.completion.chunk"]
+    created: int
+    model: str
+    choices: List[ChatCompletionChunkChoice]
+
+
+class ChatCompletionRequest(BaseModel):
+    model: str
+    messages: List[ChatMessage]
+    max_tokens: int = (64,)
+    temperature: float = (1.0,)
+    top_p: float = 1.0
+    top_k: int = 0
+    stream: bool = False
+
+
+def format_messages_as_prompt(messages: List[ChatMessage]) -> str:
+    lines: list[str] = []
+    for m in messages:
+        if m.role == "system":
+            lines.append(f"[system]\n{m.content}\n")
+        elif m.role == "user":
+            lines.append(f"User: \n{m.content}\n")
+        elif m.role == "assistant":
+            lines.append(f"Assistant: \n{m.content}\n")
+    lines.append("Assistant:")
+    return "".join(lines)
+
+
+def estimate_usage(
+    engine: InferenceEngine,
+    prompt: str,
+    completion: str,
+) -> UsageInfo:
+    tokenizer = engine.tokenizer
+    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
+    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
+    prompt_tokens = len(prompt_ids)
+    completion_tokens = len(completion_ids)
+    return UsageInfo(
+        prompt_tokens=prompt_tokens,
+        completion_tokens=completion_tokens,
+        total_tokens=prompt_tokens + completion_tokens,
+    )
+
+
 def create_app(engine: InferenceEngine) -> FastAPI:
     app = FastAPI(title="roseinfer", version="0.1.0")
 
@@ -58,11 +154,116 @@ def create_app(engine: InferenceEngine) -> FastAPI:
             temperature=body.temperature,
             top_k=body.top_k,
             top_p=body.top_p,
-            stop_on_eos=body.stop_on_eos,
-            do_sample=body.do_sample,
+            stop_on_eos=True,
+            do_sample=True,
         )
         return GenerateResponse(text=text)
 
+    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
+    def chat_completions(
+        body: ChatCompletionRequest,
+    ) -> ChatCompletionResponse | StreamingResponse:
+        prompt = format_messages_as_prompt(body.messages)
+        created = int(time.time())
+        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
+        model_name = body.model or "roseinfer"
+        if body.stream:
+
+            def event_stream():
+                first_chunk = ChatCompletionChunk(
+                    id=completion_id,
+                    object="chat.completion.chunk",
+                    created=created,
+                    model=model_name,
+                    choices=[
+                        ChatCompletionChunkChoice(
+                            index=0,
+                            delta=ChatCompletionChunkDelta(
+                                role="assistant",
+                                content="",
+                            ),
+                            finish_reason=None,
+                        )
+                    ],
+                )
+                yield f"data: {first_chunk.model_dump_json()}\n\n".encode("utf-8")
+                for piece in engine.stream_generate(
+                    prompt=prompt,
+                    max_new_tokens=body.max_tokens,
+                    temperature=body.temperature,
+                    top_k=body.top_k,
+                    top_p=body.top_p,
+                    stop_on_eos=True,
+                    do_sample=True,
+                ):
+                    if not piece:
+                        continue
+                    chunk = ChatCompletionChunk(
+                        id=completion_id,
+                        object="chat.completion.chunk",
+                        created=created,
+                        model=model_name,
+                        choices=[
+                            ChatCompletionChunkChoice(
+                                index=0,
+                                delta=ChatCompletionChunkDelta(
+                                    role=None,
+                                    content=piece,
+                                ),
+                                finish_reason=None,
+                            )
+                        ],
+                    )
+                    yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
+                final_chunk = ChatCompletionChunk(
+                    id=completion_id,
+                    object="chat.completion.chunk",
+                    created=created,
+                    model=model_name,
+                    choices=[
+                        ChatCompletionChunkChoice(
+                            index=0,
+                            delta=ChatCompletionChunkDelta(
+                                role=None,
+                                content=None,
+                            ),
+                            finish_reason="stop",
+                        )
+                    ],
+                )
+                yield f"data: {final_chunk.model_dump_json()}\n\n".encode("utf-8")
+                yield b"data: [DONE]\n\n"
+
+            return StreamingResponse(
+                event_stream(),
+                media_type="text/event-stream",
+            )
+        text = engine.generate(
+            prompt=prompt,
+            max_new_tokens=body.max_tokens,
+            temperature=body.temperature,
+            top_k=body.top_k,
+            top_p=body.top_p,
+            stop_on_eos=True,
+            do_sample=True,
+        )
+        usage = estimate_usage(engine, prompt, text)
+        resp = ChatCompletionResponse(
+            id=completion_id,
+            object="chat.completion",
+            created=created,
+            model=model_name,
+            choices=[
+                ChatCompletionChoice(
+                    index=0,
+                    message=ChatMessage(role="assistant", content=text),
+                    finish_reason="stop",
+                )
+            ],
+            usage=usage,
+        )
+        return resp
+
     return app
 
 

```



## 运行

先起服务：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ python -m rosellm.roseinfer.server --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt --tokenizer-name gpt2 --device cuda --port 8080
INFO:     Started server process [1218366]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

然后执行请求得到响应结果：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ curl -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "roseinfer-gpt2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": false
  }' | jq
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   728  100   469  100   259   1467    810 --:--:-- --:--:-- --:--:--  2282
{
  "id": "chatcmpl-94cb6aef55da44d2ba0a86cb166ae61d",
  "object": "chat.completion",
  "created": 1765256383,
  "model": "roseinfer-gpt2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "[system]\nYou are a helpful assistant.\nUser: \nHello!\nAssistant: [P]\nThe website is home to the\nregarded by the website, the\nprofit website and the\npublic domain. It is a\nservice"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 52,
    "total_tokens": 72
  }
}
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ 
```

测试流式：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ curl -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
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
data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" What"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" the"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" best"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" bet"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"?"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" best"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" bet"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" that"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" you"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" can"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" do"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" something"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" and"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" get"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" your"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" message"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" on"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" the"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" computer"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"."},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" right"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" bet"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" that"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" you"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":" are"},"finish_reason":null}]}

data: {"id":"chatcmpl-8e8d5c8033ea4e2482ab1370bde7ad0b","object":"chat.completion.chunk","created":1765256497,"model":"roseinfer-gpt2","choices":[{"index":0,"delta":{"role":null,"content":null},"finish_reason":"stop"}]}

data: [DONE]

(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ 
```

