---
classes: wide2
title: "从零实现 LLM Inference：009. Batch Decoding"
excerpt: "实现 batch decode，并隐式实现 continuous batching。"
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

本 PR 来继续完善推理框架，目前我们最大的问题是 decode 的时候是遍历每个 session，依次做的 decode，在本 PR 下，我们会把这个逻辑改成 batch decode，也就是在 decode 的那个 forward 之前，把所有 session 的 last token id 都拿出来聚合成一个 tensor 然后一块走新的 forward，结果再分拆到各个 session 上，kv-cache 等也做类似的合并分拆操作。

并且在这种改法之后，实际上隐式实现了 continuous batching，也就是可以在 decode 的时候随时加入新的 request，只不过还没有实现并发控制，必须在循环中手动同步加入新请求。

## 代码变更

### `engine.py`

最主要的实际上就是在 InferenceEngine class 上面加了一个 decode_step_sessions 方法，入参接受 sessions，内部把 sessions 提取出来 last token id 以及 kv-cache，把他们聚合后走一次 forward，forward 的结果再分拆到各 session 里。

此外就是在 Offline Scheduler class 上写了一个 step 方法，里面去调用 decode_step_sessions，然后对结果进行采样输出（额外添加了一个 apply_batch_logits 的方法，接受 logits，输出对应的采样 token）。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index ab1e640..4fbf631 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -600,6 +600,124 @@ class InferenceEngine:
         if any(tails):
             yield tails
 
+    @torch.no_grad()
+    def decode_step_sessions(
+        self,
+        sessions: list["InferenceSession"],
+    ) -> torch.Tensor:
+        assert sessions
+        from torch.amp import autocast
+
+        device = self.device
+        batch_size = len(sessions)
+        last_ids: list[int] = []
+        seq_lens: list[int] = []
+        for sess in sessions:
+            if sess.finished:
+                continue
+            assert sess.kv_cache is not None
+            assert sess.generated_ids
+            last_ids.append(sess.generated_ids[-1])
+            key0, _ = sess.kv_cache[0]
+            seq_lens.append(key0.size(2))
+        assert len(last_ids) == batch_size
+        lens = torch.tensor(seq_lens, device=device, dtype=torch.long)
+        max_len = max(seq_lens)
+        input_ids = torch.tensor(  # [B, 1]
+            last_ids,
+            dtype=torch.long,
+            device=device,
+        ).view(batch_size, 1)
+        past_mask = torch.arange(  # [B, max_len], bool
+            max_len,
+            device=device,
+        ).unsqueeze(0) < lens.unsqueeze(1)
+        new_mask = torch.ones(  # [B, 1]
+            batch_size,
+            1,
+            device=device,
+            dtype=past_mask.dtype,
+        )
+        attention_mask = torch.cat(
+            [past_mask, new_mask],
+            dim=1,
+        ).to(torch.long)
+
+        batched_past = []
+        num_layers = len(sessions[0].kv_cache)
+        for layer_idx in range(num_layers):
+            k_list = []
+            v_list = []
+            for idx, sess in enumerate(sessions):
+                k_layer, v_layer = sess.kv_cache[layer_idx]
+                T_i = seq_lens[idx]
+                if T_i < max_len:
+                    pad_len = max_len - T_i
+                    pad_shape = (
+                        1,
+                        k_layer.size(1),
+                        pad_len,
+                        k_layer.size(3),
+                    )
+                    k_pad = torch.zeros(
+                        pad_shape,
+                        dtype=k_layer.dtype,
+                        device=k_layer.device,
+                    )
+                    v_pad = torch.zeros(
+                        pad_shape,
+                        dtype=v_layer.dtype,
+                        device=v_layer.device,
+                    )
+                    k_full = torch.cat(
+                        [k_layer, k_pad],
+                        dim=2,
+                    )
+                    v_full = torch.cat(
+                        [v_layer, v_pad],
+                        dim=2,
+                    )
+                else:
+                    k_full = k_layer
+                    v_full = v_layer
+                k_list.append(k_full)
+                v_list.append(v_full)
+            k_cat = torch.cat(k_list, dim=0)
+            v_cat = torch.cat(v_list, dim=0)
+            batched_past.append((k_cat, v_cat))
+        if self.use_amp:
+            with autocast(
+                device_type=device.type,
+                dtype=self.amp_dtype,
+            ):
+                logits, _, presents = self.model(
+                    input_ids=input_ids,
+                    attention_mask=attention_mask,
+                    labels=None,
+                    past_key_values=tuple(batched_past),
+                    use_cache=True,
+                )
+        else:
+            logits, _, presents = self.model(
+                input_ids=input_ids,
+                attention_mask=attention_mask,
+                labels=None,
+                past_key_values=tuple(batched_past),
+                use_cache=True,
+            )
+        last_logits = logits[:, -1, :]  # [B, V]
+        for layer_idx in range(num_layers):
+            k_b, v_b = presents[layer_idx]
+            for idx, sess in enumerate(sessions):
+                if sess.finished:
+                    continue
+                prev_len = seq_lens[idx]
+                new_len = prev_len + 1
+                k_slice = k_b[idx : idx + 1, :, :new_len, :].contiguous()
+                v_slice = v_b[idx : idx + 1, :, :new_len, :].contiguous()
+                sess.kv_cache[layer_idx] = (k_slice, v_slice)
+        return last_logits
+
 
 class InferenceSession:
     def __init__(self, engine: "InferenceEngine") -> None:
@@ -799,6 +917,33 @@ class InferenceSession:
             self.finished = True
         return token_id
 
+    @torch.no_grad()
+    def apply_batch_logits(
+        self,
+        last_logits: torch.Tensor,
+    ) -> int | None:
+        if self.finished:
+            return None
+        eng = self.engine
+        logits_2d = last_logits.view(1, -1)  # [1, V]
+        next_token = eng._sample_next_token(
+            logits_2d,
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
     def release_kv_blocks(self) -> None:
         if self.kv_manager is None:
             return
@@ -897,20 +1042,30 @@ class OfflineScheduler:
         self._sessions[request_id] = session
         return request_id
 
+    def has_unfinished(self) -> bool:
+        return any(not sess.finished for sess in self._sessions.values())
+
+    @torch.no_grad()
+    def step(self) -> dict[int, int]:
+        active_pairs: list[tuple[int, InferenceSession]] = [
+            (rid, sess) for rid, sess in self._sessions.items() if not sess.finished
+        ]
+        if not active_pairs:
+            return {}
+        sessions = [pair[1] for pair in active_pairs]
+        last_logits = self.engine.decode_step_sessions(sessions)
+        step_tokens: dict[int, int] = {}
+        for idx, (rid, sess) in enumerate(active_pairs):
+            logits_row = last_logits[idx]
+            token_id = sess.apply_batch_logits(logits_row)
+            if token_id is not None:
+                step_tokens[rid] = token_id
+        return step_tokens
+
     @torch.no_grad()
     def run(self) -> dict[int, str]:
-        active_ids: set[int] = {
-            rid for rid, sess in self._sessions.items() if not sess.finished
-        }
-        while active_ids:
-            for rid in list(active_ids):
-                session = self._sessions[rid]
-                if session.finished:
-                    active_ids.remove(rid)
-                    continue
-                _ = session.step_once()
-                if session.finished:
-                    active_ids.remove(rid)
+        while self.has_unfinished():
+            self.step()
         outputs: dict[int, str] = {}
         for rid, session in self._sessions.items():
             outputs[rid] = session.decode_text()
```

## 运行

再把之前的 offline example 执行一下：

```shell
$ ./offline_example.sh 
### request 0
hi, Â¢:
- The figure of the possible locations in the study was that:
- The level of the population was defined by the mean-weight ratio.
- The level of the population was defined by the mean-weight ratio in the total body.
- The level of the population was defined by the mean-weight ratio.
- The overall-weight ratio was defined by the mean-weight ratio.
- The average-weight ratio was defined by the mean-weight ratio

### request 1
hello, the best suited and the control of situations.
and officials for tallening, it is very recently been evokes a thunderstorms, and the task.
Model TEACHESAYSerrillas, with the ambiguous parts for the simplestly, a thoughtfully detaileding up to BDS, the commentaries and lyrics, and insurance companies around 6thane (to-ofiscountains backbones, or the otherworldlyly date backpacks that he/hertzuniversity, but

### request 2
how is that he/or her route.
parallel data, the military bases.
Graphic description in favor of course, you feel free online. It is a series. This has been made the Australian State University of this article, and industry, it in terms requires that keeps a crime of the discovery. Thesserscentreduced from storage, which currently accepted by many times, and culture has been identified.
Azzles with a general public transport your knowledge of the term,
```



