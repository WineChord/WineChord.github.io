---
classes: wide2
title: "从零实现 LLM Inference：007. Offline Scheduler"
excerpt: "实现离线调度器，支持并发请求。"
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

在实现了基础的 generate，kv-cache，sampling，batch，streaming，session 之后，我们需要进一步向“工业化”迈进，本 PR 需要实现一个 offline scheduler，为之后的 continuous batching、paged attention 等特性做铺垫，上一个 PR 我们是从 engine 中拆出了一个单独的 session class 用来表示每个请求所对应的数据结构，但是 engine 部分还能再继续拆，他目前糅合了模型本身的静态信息以及实际的运行，我们需要把执行流程拆出来一个 scheduler，在本 PR，先实现一个最简单最基础的 scheduler，也就是能够在最开始添加所有的 prompts，从 prompts 构建 sessions，然后 schedule 的逻辑就是循环遍历这些 sessions，把每个 session 做一下一步 decode，直到所有 session 都遇到 eos。更加复杂的 schedule 以及 online schedule 等在后续 PR 会陆续涉及。



## 代码变更

### `engine.py`

主要变更就是 inference session 上面加了一些辅助函数，比如 set_generation_config, all_token_ids, decode_text 以及一个 step_once（内部会调用 decode_step + sample next token），然后就是一个单独的 OfflineScheduler class，包括 add_request 以及 run 这两个方法，前者用来添加 prompt，后者用来执行运行逻辑：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 15b8788..16cd1f8 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -593,6 +593,45 @@ class InferenceSession:
     def __init__(self, engine: "InferenceEngine") -> None:
         self.engine = engine
         self.kv_cache = None
+        self.input_ids: torch.Tensor | None = None
+        self.generated_ids: list[int] = []
+        self.finished: bool = False
+        self.max_new_tokens: int = 0
+        self.temperature: float = 1.0
+        self.top_k: int = 0
+        self.top_p: float = 1.0
+        self.do_sample: bool = False
+        self.stop_on_eos: bool = True
+        self.step_count: int = 0
+
+    def set_generation_config(
+        self,
+        max_new_tokens: int,
+        temperature: float,
+        top_k: int,
+        top_p: float,
+        do_sample: bool,
+        stop_on_eos: bool,
+    ) -> None:
+        self.max_new_tokens = max_new_tokens
+        self.temperature = temperature
+        self.top_k = top_k
+        self.top_p = top_p
+        self.do_sample = do_sample
+        self.stop_on_eos = stop_on_eos
+
+    def all_token_ids(self) -> list[int]:
+        base_ids: list[int] = []
+        if self.input_ids is not None:
+            base_ids = list(self.input_ids[0].tolist())
+        return base_ids + self.generated_ids
+
+    def decode_text(self) -> str:
+        token_ids = self.all_token_ids()
+        return self.engine.tokenizer.decode(
+            token_ids,
+            skip_special_tokens=True,
+        )
 
     @torch.no_grad()
     def prefill(
@@ -691,6 +730,33 @@ class InferenceSession:
         next_logits = logits[:, -1, :]  # [1, V]
         return next_logits  # [1, vocab]
 
+    @torch.no_grad()
+    def step_once(self) -> int | None:
+        if self.finished:
+            return None
+        if not self.generated_ids:
+            raise RuntimeError("no generated ids, call prefill first")
+        last_token_id = self.generated_ids[-1]
+        last_logits = self.decode_step(last_token_id)
+        eng = self.engine
+        next_token = eng._sample_next_token(
+            last_logits,
+            temperature=self.temperature,
+            top_k=self.top_k,
+            top_p=self.top_p,
+            do_sample=self.do_sample,
+        )
+        token_id = int(next_token)
+        self.generated_ids.append(token_id)
+        self.step_count += 1
+        if self.stop_on_eos:
+            eos_id = eng.eos_token_id
+            if eos_id is not None and token_id == eos_id:
+                self.finished = True
+        if self.max_new_tokens > 0 and self.step_count >= self.max_new_tokens:
+            self.finished = True
+        return token_id
+
     @torch.no_grad()
     def decode_step_batch(
         self,
@@ -724,3 +790,77 @@ class InferenceSession:
         self.kv_cache = presents
         next_logits = logits[:, -1, :]  # [B, V]
         return next_logits  # [B, V]
+
+
+class OfflineScheduler:
+    def __init__(self, engine: "InferenceEngine") -> None:
+        self.engine = engine
+        self._sessions: dict[int, InferenceSession] = {}
+        self._next_request_id: int = 0
+
+    @torch.no_grad()
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
+        eng = self.engine
+        eng.model.eval()
+        input_ids = eng._encode_prompt(prompt)  # [1, T0]
+        input_ids = eng._maybe_truncate(input_ids)  # [1, T]
+        session = InferenceSession(eng)
+        session.input_ids = input_ids
+        session.set_generation_config(
+            max_new_tokens=max_new_tokens,
+            temperature=temperature,
+            top_k=top_k,
+            top_p=top_p,
+            do_sample=do_sample,
+            stop_on_eos=stop_on_eos,
+        )
+        logits = session.prefill(input_ids)  # [1, T, V]
+        last_logits = logits[:, -1, :]  # [1, V]
+        next_token = eng._sample_next_token(
+            last_logits,
+            temperature=temperature,
+            top_k=top_k,
+            top_p=top_p,
+            do_sample=do_sample,
+        )
+        token_id = int(next_token)
+        session.generated_ids.append(token_id)
+        session.step_count = 1
+        if stop_on_eos:
+            eos_id = eng.eos_token_id
+            if eos_id is not None and token_id == eos_id:
+                session.finished = True
+        if max_new_tokens > 0 and session.step_count >= max_new_tokens:
+            session.finished = True
+        request_id = self._next_request_id
+        self._next_request_id += 1
+        self._sessions[request_id] = session
+        return request_id
+
+    @torch.no_grad()
+    def run(self) -> dict[int, str]:
+        active_ids: set[int] = {
+            rid for rid, sess in self._sessions.items() if not sess.finished
+        }
+        while active_ids:
+            for rid in list(active_ids):
+                session = self._sessions[rid]
+                if session.finished:
+                    active_ids.remove(rid)
+                    continue
+                _ = session.step_once()
+                if session.finished:
+                    active_ids.remove(rid)
+        outputs: dict[int, str] = {}
+        for rid, session in self._sessions.items():
+            outputs[rid] = session.decode_text()
+        return outputs

```





## 运行

运行下 offline example 看看：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ cat offline_example.sh 
#!/bin/bash

python -m roseinfer.offline_example \
  --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt \
  --tokenizer-name gpt2 \
  --prompts "hi, " "hello," "how" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95 \
  --do-sample

(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ ./offline_example.sh 
### request 0
hi, !!!
-per is about the source of the development of the endical �

### request 1
hello, after the eastern and the province--report, this space the visit the way

### request 2
how, and experience the clothing, and will be used and
Transcompliance.

```

