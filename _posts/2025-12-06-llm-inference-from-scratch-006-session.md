---
classes: wide2
title: "从零实现 LLM Inference：006. Inference Session"
excerpt: "添加 Inference Session，支持并发请求。"
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

本 PR 来做一些工业级别的重构，主要是添加一下 Inference Session，在我们的当前实现里面，只有一个 Inference Engine class，kv-cache 的管理，model，prefill，decode，generate 逻辑全部都在这个 engine 当中，最严重的问题是我们没有办法用这个 engine class 处理并发请求，因为这个 engine class 只能绑定一个 kv-cache，所以我们这个 PR 需要把请求相关的东西拆出来，拆到 Inference Session class 中，主要是 kv-cache，prefill，decode 等这些逻辑。

## 代码变更

### `engine.py`

看上去变更很多，实际上就是把 prefill, prefill_batch, decode_one_step, decode_one_step_batch 给放到了 InferenceSession class 中，然后把一些 self 改成了 engine 而已：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 47ec839..15b8788 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -64,7 +64,6 @@ class InferenceEngine:
 
         self._make_detok = make_detok
 
-        self.kv_cache = None
         if self.config.vocab_size < self.tokenizer.vocab_size:
             raise ValueError("the model vocab_size is less than tokenizer vocab_size")
 
@@ -216,133 +215,6 @@ class InferenceEngine:
             device=self.device,
         )
 
-    @torch.no_grad()
-    def prefill(
-        self,
-        prompt_ids: torch.Tensor,  # [..., T0]
-    ):
-        from torch.amp import autocast
-
-        input_ids = self._maybe_truncate(prompt_ids)
-        if self.use_amp:
-            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
-                logits, _, presents = self.model(
-                    input_ids=input_ids,
-                    attention_mask=None,
-                    labels=None,
-                    past_key_values=None,
-                    use_cache=True,
-                )
-        else:
-            logits, _, presents = self.model(
-                input_ids=input_ids,
-                attention_mask=None,
-                labels=None,
-                past_key_values=None,
-                use_cache=True,
-            )
-        self.kv_cache = presents
-        return logits  # [..., T0, vocab]
-
-    @torch.no_grad()
-    def prefill_batch(
-        self,
-        input_ids: torch.Tensor,
-        attention_mask: Optional[torch.Tensor] = None,
-    ) -> torch.Tensor:
-        from torch.amp import autocast
-
-        input_ids = self._maybe_truncate(input_ids)
-        if attention_mask is not None and input_ids.size(1) < attention_mask.size(1):
-            attention_mask = attention_mask[:, -input_ids.size(1) :]
-        if self.use_amp:
-            with autocast(
-                device_type=self.device.type,
-                dtype=self.amp_dtype,
-            ):
-                logits, _, presents = self.model(
-                    input_ids=input_ids,
-                    attention_mask=attention_mask,
-                    labels=None,
-                    past_key_values=None,
-                    use_cache=True,
-                )
-        else:
-            logits, _, presents = self.model(
-                input_ids=input_ids,
-                attention_mask=attention_mask,
-                labels=None,
-                past_key_values=None,
-                use_cache=True,
-            )
-        self.kv_cache = presents
-        last_logits = logits[:, -1, :]  # [batch, vocab]
-        return last_logits
-
-    @torch.no_grad()
-    def decode_step(self, last_token_id: int) -> torch.Tensor:
-        assert self.kv_cache is not None
-        from torch.amp import autocast
-
-        input_ids = torch.tensor(  # [1, 1]
-            [[last_token_id]],
-            dtype=torch.long,
-            device=self.device,
-        )
-        if self.use_amp:
-            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
-                logits, _, presents = self.model(
-                    input_ids=input_ids,
-                    attention_mask=None,
-                    labels=None,
-                    past_key_values=self.kv_cache,
-                    use_cache=True,
-                )
-        else:
-            logits, _, presents = self.model(
-                input_ids=input_ids,
-                attention_mask=None,
-                labels=None,
-                past_key_values=self.kv_cache,
-                use_cache=True,
-            )
-        self.kv_cache = presents
-        next_logits = logits[:, -1, :]  # [1, V]
-        return next_logits  # [1, vocab]
-
-    @torch.no_grad()
-    def decode_step_batch(
-        self,
-        last_token_ids: torch.Tensor,
-    ) -> torch.Tensor:
-        assert self.kv_cache is not None
-        from torch.amp import autocast
-
-        input_ids = last_token_ids.view(-1, 1)  # [B, 1]
-        if self.use_amp:
-            with autocast(
-                device_type=self.device.type,
-                dtype=self.amp_dtype,
-            ):
-                logits, _, presents = self.model(
-                    input_ids=input_ids,
-                    attention_mask=None,
-                    labels=None,
-                    past_key_values=self.kv_cache,
-                    use_cache=True,
-                )
-        else:
-            logits, _, presents = self.model(
-                input_ids=input_ids,
-                attention_mask=None,
-                labels=None,
-                past_key_values=self.kv_cache,
-                use_cache=True,
-            )
-        self.kv_cache = presents
-        next_logits = logits[:, -1, :]  # [B, V]
-        return next_logits  # [B, V]
-
     @torch.no_grad()
     def generate(
         self,
@@ -355,9 +227,10 @@ class InferenceEngine:
         do_sample: bool = False,
     ) -> str:
         self.model.eval()
+        session = InferenceSession(self)
         input_ids = self._encode_prompt(prompt)  # [1, T0]
         input_ids = self._maybe_truncate(input_ids)  # [1, T]
-        logits = self.prefill(input_ids)  # [1, T, V]
+        logits = session.prefill(input_ids)  # [1, T, V]
         last_logits = logits[:, -1, :]  # [1, V]
         generated_ids = input_ids[0].tolist()
         if max_new_tokens <= 0:
@@ -389,7 +262,7 @@ class InferenceEngine:
             return self._decode_tokens(generated)
 
         for _ in range(max_new_tokens - 1):
-            next_logits = self.decode_step(last_token_id)
+            next_logits = session.decode_step(last_token_id)
             next_id = self._sample_next_token(
                 logits=next_logits,
                 temperature=temperature,
@@ -426,9 +299,10 @@ class InferenceEngine:
     ) -> list[str]:
         assert len(prompts) > 0
         self.model.eval()
+        session = InferenceSession(self)
         input_ids, attn_mask = self._encode_prompts_batch(prompts)
         batch_size = input_ids.size(0)
-        last_logits = self.prefill_batch(
+        last_logits = session.prefill_batch(
             input_ids,
             attention_mask=attn_mask,
         )
@@ -473,7 +347,7 @@ class InferenceEngine:
                 and all(pos is not None for pos in eos_positions)
             ):
                 break
-            next_logits = self.decode_step_batch(last_token_ids)
+            next_logits = session.decode_step_batch(last_token_ids)
             next_ids = self._sample_next_token_batch(
                 logits=next_logits,
                 temperature=temperature,
@@ -525,6 +399,7 @@ class InferenceEngine:
         do_sample: bool = False,
     ) -> Iterator[str]:
         self.model.eval()
+        session = InferenceSession(self)
         token_ids = self.tokenizer.encode(
             prompt,
             add_special_tokens=False,
@@ -538,7 +413,7 @@ class InferenceEngine:
         )
         detok = self._make_detok()
         detok.start_prompt(token_ids)
-        prefill_logits = self.prefill(ids_tensor)  # [1, T, V]
+        prefill_logits = session.prefill(ids_tensor)  # [1, T, V]
         last_logits = prefill_logits[:, -1, :]  # [1, V]
         if max_new_tokens <= 0:
             piece = detok.flush()
@@ -566,7 +441,7 @@ class InferenceEngine:
             return
         last_token_id = next_id
         for _ in range(max_new_tokens - 1):
-            next_logits = self.decode_step(last_token_id)  # [1, V]
+            next_logits = session.decode_step(last_token_id)  # [1, V]
             next_id = self._sample_next_token(
                 logits=next_logits,
                 temperature=temperature,
@@ -600,6 +475,7 @@ class InferenceEngine:
         do_sample: bool = True,
     ) -> Iterator[list[str]]:
         self.model.eval()
+        session = InferenceSession(self)
         batch_size = len(prompts)
         if batch_size == 0:
             return
@@ -635,7 +511,7 @@ class InferenceEngine:
             dtype=torch.long,
             device=self.device,
         )
-        last_logits = self.prefill_batch(
+        last_logits = session.prefill_batch(
             input_ids,
             attention_mask=attention_mask,
         )  # [B, V]
@@ -674,7 +550,7 @@ class InferenceEngine:
             device=self.device,
         )
         for _ in range(max_new_tokens - 1):
-            next_logits = self.decode_step_batch(last_token_ids)
+            next_logits = session.decode_step_batch(last_token_ids)
             new_ids: list[int] = []
             pieces: list[str] = []
             for b in range(batch_size):
@@ -711,3 +587,140 @@ class InferenceEngine:
             tails.append(tail)
         if any(tails):
             yield tails
+
+
+class InferenceSession:
+    def __init__(self, engine: "InferenceEngine") -> None:
+        self.engine = engine
+        self.kv_cache = None
+
+    @torch.no_grad()
+    def prefill(
+        self,
+        prompt_ids: torch.Tensor,  # [..., T0]
+    ):
+        from torch.amp import autocast
+
+        eng = self.engine
+        input_ids = eng._maybe_truncate(prompt_ids)
+        if eng.use_amp:
+            with autocast(device_type=eng.device.type, dtype=eng.amp_dtype):
+                logits, _, presents = eng.model(
+                    input_ids=input_ids,
+                    attention_mask=None,
+                    labels=None,
+                    past_key_values=None,
+                    use_cache=True,
+                )
+        else:
+            logits, _, presents = eng.model(
+                input_ids=input_ids,
+                attention_mask=None,
+                labels=None,
+                past_key_values=None,
+                use_cache=True,
+            )
+        self.kv_cache = presents
+        return logits  # [..., T0, vocab]
+
+    @torch.no_grad()
+    def prefill_batch(
+        self,
+        input_ids: torch.Tensor,
+        attention_mask: Optional[torch.Tensor] = None,
+    ) -> torch.Tensor:
+        from torch.amp import autocast
+
+        eng = self.engine
+        input_ids = eng._maybe_truncate(input_ids)
+        if attention_mask is not None and input_ids.size(1) < attention_mask.size(1):
+            attention_mask = attention_mask[:, -input_ids.size(1) :]
+        if eng.use_amp:
+            with autocast(
+                device_type=eng.device.type,
+                dtype=eng.amp_dtype,
+            ):
+                logits, _, presents = eng.model(
+                    input_ids=input_ids,
+                    attention_mask=attention_mask,
+                    labels=None,
+                    past_key_values=None,
+                    use_cache=True,
+                )
+        else:
+            logits, _, presents = eng.model(
+                input_ids=input_ids,
+                attention_mask=attention_mask,
+                labels=None,
+                past_key_values=None,
+                use_cache=True,
+            )
+        self.kv_cache = presents
+        last_logits = logits[:, -1, :]  # [batch, vocab]
+        return last_logits
+
+    @torch.no_grad()
+    def decode_step(self, last_token_id: int) -> torch.Tensor:
+        assert self.kv_cache is not None
+        from torch.amp import autocast
+
+        eng = self.engine
+        input_ids = torch.tensor(  # [1, 1]
+            [[last_token_id]],
+            dtype=torch.long,
+            device=eng.device,
+        )
+        if eng.use_amp:
+            with autocast(device_type=eng.device.type, dtype=eng.amp_dtype):
+                logits, _, presents = eng.model(
+                    input_ids=input_ids,
+                    attention_mask=None,
+                    labels=None,
+                    past_key_values=self.kv_cache,
+                    use_cache=True,
+                )
+        else:
+            logits, _, presents = eng.model(
+                input_ids=input_ids,
+                attention_mask=None,
+                labels=None,
+                past_key_values=self.kv_cache,
+                use_cache=True,
+            )
+        self.kv_cache = presents
+        next_logits = logits[:, -1, :]  # [1, V]
+        return next_logits  # [1, vocab]
+
+    @torch.no_grad()
+    def decode_step_batch(
+        self,
+        last_token_ids: torch.Tensor,
+    ) -> torch.Tensor:
+        assert self.kv_cache is not None
+        from torch.amp import autocast
+
+        eng = self.engine
+        input_ids = last_token_ids.view(-1, 1)  # [B, 1]
+        if eng.use_amp:
+            with autocast(
+                device_type=eng.device.type,
+                dtype=eng.amp_dtype,
+            ):
+                logits, _, presents = eng.model(
+                    input_ids=input_ids,
+                    attention_mask=None,
+                    labels=None,
+                    past_key_values=self.kv_cache,
+                    use_cache=True,
+                )
+        else:
+            logits, _, presents = eng.model(
+                input_ids=input_ids,
+                attention_mask=None,
+                labels=None,
+                past_key_values=self.kv_cache,
+                use_cache=True,
+            )
+        self.kv_cache = presents
+        next_logits = logits[:, -1, :]  # [B, V]
+        return next_logits  # [B, V]

```

## 运行

把之前能运行的命令重新运行下，保证结果没问题就行：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ cat generate.sh 
#!/bin/bash

python -m roseinfer.cli_generate --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt --tokenizer-name gpt2 --max-new-tokens 1000 --stream --prompt "hi, " --top-k 40 --top-p 0.99 --do-sample
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ ./generate.sh 
[roseinfer] device: cuda
[roseinfer] use_amp: True
[roseinfer] prompt: hi, 
[roseinfer] streaming output: Â on.
P.
”.
One of the future-c. The first a place and the water.
The early in the the world a long and that they also the way, in the process to work of our health issues are important.
In the “the power. On the two or the time and the "A recent long that they had the world a big way”, it were no good of other in the early years to be a year. There has been involved in the process as well in the last year, a great work at them from the time.
1.
If we are the future people who can be made of the life or the United States. The "I did the world.’s”.
The "I. A: You will work by a “" was an unusual of a few to make a high.
Topal-1/L) and all to the world and the other in the health are found that have a single, you find a year, an effective a man to the last year. For a man, a good, and all you have made of a year, a common are it were also in the world, we are the two children at the first is a new world.
" has been found that the same process of the first.
D; (3)
What?
D is as a major effects of the world of a year years,
The same species. A.
This in high risk-13.
In the the last years to be a way from the water (13 This is a more for the
The main types of the history of its name is to the world of the first to the future.
S. A number of the most high work for an energy use with the end with all, all other countries to address the world as of the first and the “T.
The American
A”, and the best-3, and the world-11012-ft (5.
When the
There are a child and a new to the world, and all are no long-1, the most-B., and other people and,.
What”; it are that is a way to use of a “In the people of the end, the health is about the long-D for the world- The way to provide all a more on the other, that the process. On its, that the way.
-
The new to help to the two year. When the first-18,’s.
B and that it. The first the use to get out that a few days.
It were used to work of all in the people to the use, he made about how the end is the main size for a very low in the first on a new research the last year to have the first a one, of your experience and use in his political and you will be able to the way it will be not to the most of the state in a world is one, ‘C in a way’�s in the use is so with the day, and is a time to be able to avoid a "t the “There had had a time or a way you are not a day.
A
T
This
S, and that the the end a common of the use
It is used of their the day a few years. This (5, and.
For for a good for the number, we get their health of a year, the people, which can also also be well if we know the most of those of the development, and the water and a place.
In the same, the first that, when the health of a short, they need and the life, and use, is the way. There are more is a well in the same in the end is just a time on the best, which may be not to a year: The results of the first to the two species, the
-20-S.’ and it are ‘Mm.,:
The first the "D,
-The article, and the first and the world, this way to the first an environment, to a "�re on the world, a person and be so well that there are the the world. When you learn how the U. The other types of all in a short-
The "Mor of the most to be also be good“The
" (13, ‘I think to an alternative on the use can be used,, but in the environment. To learn to be the people are just that of other.
In the most large water or other the new business was also help for the end and more than an effect
The new health of "The first time of the
The-
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ 
```

