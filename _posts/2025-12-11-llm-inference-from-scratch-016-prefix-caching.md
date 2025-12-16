---
classes: wide2
title: "从零实现 LLM Inference：016. Simple Prefix Caching"
excerpt: "实现简单的 prefix caching，通过 prefix cache 来复用之前的 kv-cache。"
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

在实现了对 online/offline scheduler 的 benchmark 后，我们可以顺势做一些优化来看一些具体真实的性能提升，我们从 Prefix Caching 开始做起。

所谓 Prefix Caching，就是当我拿到用户输入的 prompt 时，我发现他的一个前缀实际上在之前我已经计算过对应的 kv-cache 了，所以对于这一部分，我们可以跳过他们的 prefill，在工业界实现中，会根据 kv block 来选这样的重复前缀做管理，更加精细化的操作是做 radix tree，可以达到 token level 的 kv-cache 复用（比如 sglang 有这些优化），我们第一个有关 prefix caching 的 PR 依然做一个最简单的方案，直接拿用户的 prompt 本身作为 key 去保存他的 kv-cache，进一步的优化留给后续 PR。

## 代码变更

主要代码变更集中在 engine.py 里面，需要添加 prefix cache 和 prefix cache entry，在 prefill 之后把 kv cache 加到 prefix cache 中或者进行复用，复用的时候需要增加引用计数，最后不使用则减掉引用计数，当 cache 内的元素过多时则用 LRU 算法来踢掉多余元素。

### `engine.py`

```python
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 68e085f..19438b0 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1,3 +1,4 @@
+from collections import OrderedDict
 from typing import Iterator, NamedTuple, Optional
 
 import torch
@@ -25,6 +26,8 @@ class InferenceEngine:
         use_amp: bool = True,
         max_position_embeddings: Optional[int] = None,
         bf16: bool = False,
+        kv_cache_max_concurrency: int = 256,
+        prefix_cache_max_entries: int = 256,
     ) -> None:
         super().__init__()
         if device is None:
@@ -65,7 +68,9 @@ class InferenceEngine:
         self._make_detok = make_detok
         block_size = 64
         max_context = max_position_embeddings or self.config.max_position_embeddings
-        max_blocks_per_layer = (max_context + block_size - 1) // block_size
+        max_concurrency = max(1, kv_cache_max_concurrency)
+        max_total_tokens = max_context * max_concurrency
+        max_blocks_per_layer = (max_total_tokens + block_size - 1) // block_size
         self.kv_manager = KVBlockManager(
             num_layers=self.config.n_layers,
             num_heads=self.config.n_heads,
@@ -75,6 +80,10 @@ class InferenceEngine:
             device=self.device,
             dtype=self.amp_dtype if self.use_amp else self.model.dtype,
         )
+        self.prefix_cache = PrefixCache(
+            self.kv_manager,
+            max_entries=prefix_cache_max_entries,
+        )
 
         if self.config.vocab_size < self.tokenizer.vocab_size:
             raise ValueError("the model vocab_size is less than tokenizer vocab_size")
@@ -138,6 +147,58 @@ class InferenceEngine:
             input_ids = input_ids[:, -max_pos:]
         return input_ids
 
+    def _maybe_prefill_with_prefix_cache(
+        self,
+        session: "InferenceSession",
+        prompt: str,
+        use_prefix_cache: bool,
+        max_new_tokens: int,
+        temperature: float,
+        top_k: int,
+        top_p: float,
+        do_sample: bool,
+        stop_on_eos: bool,
+    ) -> None:
+        input_ids = self._encode_prompt(prompt)
+        input_ids = self._maybe_truncate(input_ids)
+        session.input_ids = input_ids
+        session.set_generation_config(
+            max_new_tokens=max_new_tokens,
+            temperature=temperature,
+            top_k=top_k,
+            top_p=top_p,
+            do_sample=do_sample,
+            stop_on_eos=stop_on_eos,
+        )
+        cached_logits = None
+        if use_prefix_cache:
+            cached_logits = self.prefix_cache.attach(prompt, session)
+        if cached_logits is None:
+            logits = session.prefill(input_ids)
+            last_logits = logits[:, -1, :]
+            session.kv_cache = None
+            if use_prefix_cache:
+                self.prefix_cache.put(prompt, session, last_logits)
+        else:
+            last_logits = cached_logits
+
+        next_token = self._sample_next_token(
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
+            eos_id = self.eos_token_id
+            if eos_id is not None and token_id == eos_id:
+                session.finished = True
+        if max_new_tokens > 0 and session.step_count >= max_new_tokens:
+            session.finished = True
+
     def _top_k_logits(
         self,
         logits: torch.Tensor,  # [..., vocab]
@@ -980,6 +1041,7 @@ class InferenceSession:
         return token_id
 
     def release_kv_blocks(self) -> None:
+        self.kv_cache = None
         if self.kv_manager is None:
             return
         for layer_idx, block_ids in enumerate(self.block_ids_per_layer):
@@ -1023,9 +1085,101 @@ class InferenceSession:
         return next_logits  # [B, V]
 
 
+class PrefixCacheEntry:
+    def __init__(
+        self,
+        prompt: str,
+        prompt_length: int,
+        blocks_ids_per_layer: list[list[int]],
+        last_logits: torch.Tensor,
+    ) -> None:
+        self.prompt = prompt
+        self.prompt_length = int(prompt_length)
+        self.blocks_ids_per_layer = [list(ids) for ids in blocks_ids_per_layer]
+        self.last_logits = last_logits.detach().to("cpu")
+
+
+class PrefixCache:
+    def __init__(
+        self,
+        kv_manager: "KVBlockManager",
+        max_entries: int = 256,
+    ) -> None:
+        self.kv_manager = kv_manager
+        self.max_entries = max(0, int(max_entries))
+        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
+
+    def _release_entry(self, entry: PrefixCacheEntry) -> None:
+        for layer_idx, block_ids in enumerate(entry.block_ids_per_layer):
+            if block_ids:
+                self.kv_manager.free_blocks(layer_idx, block_ids)
+
+    def _evict_one(self) -> None:
+        if not self._entries:
+            return
+        _, entry = self._entries.popitem(last=False)
+        self._release_entry(entry)
+
+    def get(self, prompt: str) -> PrefixCacheEntry | None:
+        return self._entries.get(prompt)
+
+    def put(
+        self,
+        prompt: str,
+        session: "InferenceSession",
+        last_logits: torch.Tensor,
+    ) -> None:
+        if prompt in self._entries:
+            self._entries.move_to_end(prompt)
+            return
+        if session.kv_manager is None:
+            return
+        prompt_length = session.prompt_length
+        block_ids_per_layer = [list(ids) for ids in session.block_ids_per_layer]
+        for block_ids in block_ids_per_layer:
+            if not block_ids:
+                continue
+            self.kv_manager.incref_blocks(block_ids)
+        entry = PrefixCacheEntry(
+            prompt=prompt,
+            prompt_length=prompt_length,
+            blocks_ids_per_layer=block_ids_per_layer,
+            last_logits=last_logits,
+        )
+        while self.max_entries > 0 and len(self._entries) >= self.max_entries:
+            self._evict_one()
+        self._entries[prompt] = entry
+        self._entries.move_to_end(prompt)
+
+    def attach(
+        self,
+        prompt: str,
+        session: "InferenceSession",
+    ) -> torch.Tensor | None:
+        entry = self._entries.get(prompt)
+        if entry is None:
+            return None
+        self._entries.move_to_end(prompt)
+        session.prompt_length = entry.prompt_length
+        session.block_ids_per_layer = []
+        for block_ids in entry.blocks_ids_per_layer:
+            if not block_ids:
+                session.block_ids_per_layer.append([])
+                continue
+            self.kv_manager.incref_blocks(block_ids)
+            session.block_ids_per_layer.append(list(block_ids))
+        last_logits = entry.last_logits.to(session.engine.device)
+        return last_logits
+
+
 class OfflineScheduler:
-    def __init__(self, engine: "InferenceEngine") -> None:
+    def __init__(
+        self,
+        engine: "InferenceEngine",
+        use_prefix_cache: bool = True,
+    ) -> None:
         self.engine = engine
+        self.use_prefix_cache = use_prefix_cache
         self._sessions: dict[int, InferenceSession] = {}
         self._next_request_id: int = 0
 
@@ -1042,11 +1196,12 @@ class OfflineScheduler:
     ) -> int:
         eng = self.engine
         eng.model.eval()
-        input_ids = eng._encode_prompt(prompt)  # [1, T0]
-        input_ids = eng._maybe_truncate(input_ids)  # [1, T]
         session = InferenceSession(eng)
-        session.input_ids = input_ids
-        session.set_generation_config(
+
+        eng._maybe_prefill_with_prefix_cache(
+            session=session,
+            prompt=prompt,
+            use_prefix_cache=self.use_prefix_cache,
             max_new_tokens=max_new_tokens,
             temperature=temperature,
             top_k=top_k,
@@ -1054,24 +1209,6 @@ class OfflineScheduler:
             do_sample=do_sample,
             stop_on_eos=stop_on_eos,
         )
-        logits = session.prefill(input_ids)  # [1, T, V]
-        last_logits = logits[:, -1, :]  # [1, V]
-        next_token = eng._sample_next_token(
-            last_logits,
-            temperature=temperature,
-            top_k=top_k,
-            top_p=top_p,
-            do_sample=do_sample,
-        )
-        token_id = int(next_token)
-        session.generated_ids.append(token_id)
-        session.step_count = 1
-        if stop_on_eos:
-            eos_id = eng.eos_token_id
-            if eos_id is not None and token_id == eos_id:
-                session.finished = True
-        if max_new_tokens > 0 and session.step_count >= max_new_tokens:
-            session.finished = True
         request_id = self._next_request_id
         self._next_request_id += 1
         self._sessions[request_id] = session
@@ -1114,9 +1251,11 @@ class OnlineScheduler:
         self,
         engine: "InferenceEngine",
         max_batch_size: int = 8,
+        use_prefix_cache: bool = True,
     ) -> None:
         self.engine = engine
         self.max_batch_size = max_batch_size
+        self.use_prefix_cache = use_prefix_cache
         self._sessions: dict[int, InferenceSession] = {}
         self._next_request_id: int = 0
         self._round_robin_pos: int = 0
@@ -1134,11 +1273,11 @@ class OnlineScheduler:
     ) -> int:
         eng = self.engine
         eng.model.eval()
-        input_ids = eng._encode_prompt(prompt)  # [1, T0]
-        input_ids = eng._maybe_truncate(input_ids)  # [1, T]
         session = InferenceSession(eng)
-        session.input_ids = input_ids
-        session.set_generation_config(
+        eng._maybe_prefill_with_prefix_cache(
+            session=session,
+            prompt=prompt,
+            use_prefix_cache=self.use_prefix_cache,
             max_new_tokens=max_new_tokens,
             temperature=temperature,
             top_k=top_k,
@@ -1146,24 +1285,8 @@ class OnlineScheduler:
             do_sample=do_sample,
             stop_on_eos=stop_on_eos,
         )
-        logits = session.prefill(input_ids)  # [1, T, V]
-        last_logits = logits[:, -1, :]  # [1, V]
-        next_token = eng._sample_next_token(
-            last_logits,
-            temperature=temperature,
-            top_k=top_k,
-            top_p=top_p,
-            do_sample=do_sample,
-        )
-        token_id = int(next_token)
-        session.generated_ids.append(token_id)
-        session.step_count = 1
-        if stop_on_eos:
-            eos_id = eng.eos_token_id
-            if eos_id is not None and token_id == eos_id:
-                session.finished = True
-        if max_new_tokens > 0 and session.step_count >= max_new_tokens:
-            session.finished = True
+        if session.finished:
+            session.release_kv_blocks()
         request_id = self._next_request_id
         self._next_request_id += 1
         self._sessions[request_id] = session
@@ -1209,6 +1332,7 @@ class OnlineScheduler:
 
     def pop_response(self, request_id: int) -> str:
         session = self._sessions.pop(request_id)
+        session.release_kv_blocks()
         return session.decode_text()
 
 
@@ -1244,6 +1368,7 @@ class KVBlockManager:
             int,
             tuple[torch.Tensor, torch.Tensor],
         ] = {}  # global_id -> (key_block, value_block)
+        self._block_refcounts: dict[int, int] = {}
 
     def _alloc_block_index(self, layer_idx: int) -> int:
         free_list = self._free_block_indices[layer_idx]
@@ -1304,20 +1429,41 @@ class KVBlockManager:
             k_block[:, :length, :] = k_slice[0]
             v_block[:, :length, :] = v_slice[0]
             self._block_storage[global_id] = (k_block, v_block)
+            self._block_refcounts[global_id] = 1
             block_ids.append(global_id)
         return block_ids
 
+    def incref_blocks(
+        self,
+        block_ids: list[int],
+    ) -> None:
+        for global_id in block_ids:
+            self._block_refcounts[global_id] = (
+                self._block_refcounts.get(
+                    global_id,
+                    0,
+                )
+                + 1
+            )
+
     def free_blocks(
         self,
         layer_idx: int,
         block_ids: list[int],
     ) -> None:
         for global_id in block_ids:
+            ref = self._block_refcounts.get(global_id)
+            if ref is None:
+                continue
+            ref -= 1
+            if ref > 0:
+                self._block_refcounts[global_id] = ref
+                continue
+            self._block_refcounts.pop(global_id, None)
             info = self._block_infos.pop(global_id, None)
             if info is None:
                 continue
-            if info.layer != layer_idx:
-                continue
+            assert info.layer == layer_idx
             self._free_block_indices[layer_idx].append(
                 info.block_index,
             )
@@ -1356,19 +1502,39 @@ class KVBlockManager:
             )
             v_block = torch.zeros_like(k_block)
             self._block_storage[global_id] = (k_block, v_block)
+            self._block_refcounts[global_id] = 1
             block_ids.append(global_id)
         last_id = block_ids[-1]
         info = self._block_infos[last_id]
-        k_block, v_block = self._block_storage[last_id]
-        if info.length >= self.block_size:
+        ref = self._block_refcounts.get(last_id, 1)
+        if ref > 1 and info.length < self.block_size:
+            self._block_refcounts[last_id] = ref - 1
             block_idx = self._alloc_block_index(layer_idx)
-            global_id = self._to_global_block_id(
+            new_global_id = self._to_global_block_id(
                 layer_idx,
                 block_idx,
             )
-            info = KVBlockInfo(
+            new_info = KVBlockInfo(
                 layer=layer_idx,
                 block_index=block_idx,
+                start=info.start,
+                length=info.length,
+            )
+            self._block_infos[new_global_id] = new_info
+            k_block_old, v_block_old = self._block_storage[last_id]
+            k_block = k_block_old.clone()
+            v_block = v_block_old.clone()
+            self._block_storage[new_global_id] = (k_block, v_block)
+            self._block_refcounts[new_global_id] = 1
+            block_ids[-1] = new_global_id
+            last_id = new_global_id
+            info = new_info
+        if info.length >= self.block_size:
+            block_idx = self._alloc_block_index(layer_idx)
+            global_id = self._to_global_block_id(layer_idx, block_idx)
+            info = KVBlockInfo(
+                layer=info.layer,
+                block_index=block_idx,
                 start=info.start + info.length,
                 length=0,
             )
@@ -1384,6 +1550,7 @@ class KVBlockManager:
             )
             v_block = torch.zeros_like(k_block)
             self._block_storage[global_id] = (k_block, v_block)
+            self._block_refcounts[global_id] = 1
             block_ids.append(global_id)
             last_id = global_id
         info = self._block_infos[last_id]

```





## 运行

运行可以看到在使用了 prefix caching 之后，prefill 的时间有所减少，因为 prompt 可以复用之前的 kv block：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ ./benchmark_scheduler.sh  && ./benchmark_scheduler_no_prefix_cache.sh 
benchmarking scheduler with prefix cache
=== naive ===
Requests: 64
Elapsed: 4.221361 seconds
Prompt tokens: 54784
Completion tokens: 1024
Total tokens: 55808
Throughput (completion): 242.58 tokens/s
Throughput (total): 13220.38 tokens/s

=== offline ===
Requests: 64
Elapsed (prefill/add): 0.251266 seconds
Elapsed (decode/run): 3.046476 seconds
Elapsed (total): 3.297742 seconds
Prompt tokens: 54784
Completion tokens: 1024
Total tokens: 55808
Throughput (completion): 310.52 tokens/s
Throughput (total): 16923.09 tokens/s

=== online ===
Requests: 64
Elapsed (prefill/add): 0.246590 seconds
Elapsed (decode/run): 3.304484 seconds
Elapsed (total): 3.551074 seconds
Prompt tokens: 54784
Completion tokens: 1024
Total tokens: 55808
Throughput (completion): 288.36 tokens/s
Throughput (total): 15715.81 tokens/s

benchmarking scheduler without prefix cache
=== naive ===
Requests: 64
Elapsed: 4.195738 seconds
Prompt tokens: 54784
Completion tokens: 1024
Total tokens: 55808
Throughput (completion): 244.06 tokens/s
Throughput (total): 13301.12 tokens/s

=== offline ===
Requests: 64
Elapsed (prefill/add): 1.086752 seconds
Elapsed (decode/run): 3.061303 seconds
Elapsed (total): 4.148056 seconds
Prompt tokens: 54784
Completion tokens: 1024
Total tokens: 55808
Throughput (completion): 246.86 tokens/s
Throughput (total): 13454.01 tokens/s

=== online ===
Requests: 64
Elapsed (prefill/add): 1.084652 seconds
Elapsed (decode/run): 3.459252 seconds
Elapsed (total): 4.543904 seconds
Prompt tokens: 54784
Completion tokens: 1024
Total tokens: 55808
Throughput (completion): 225.36 tokens/s
Throughput (total): 12281.95 tokens/s
```

