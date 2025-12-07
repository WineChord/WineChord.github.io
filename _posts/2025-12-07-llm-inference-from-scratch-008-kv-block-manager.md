---
classes: wide2
title: "从零实现 LLM Inference：008. KV Block Manager"
excerpt: "实现基础的 kv block manager"
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

在实现了 offline scheduler  之后，我们可以往 paged attention 迈一小步，可以先实现一个 python 版本的 kv block manager，并为了使单次变更比较小，本次仅对 kv-cache 记录一下 kv block 的 meta data，暂时不在实际的 forward 使用。

## 代码变更

### `engine.py`

最主要添加的就是 kv block manager，最主要是构造一个 global id 到 kv block info 的映射，kv block info 则包含 layer index，block index，start，length 这些 metadata，然后在每个 inference session 维护一个 block ids per layer，从 layer index 映射到这个 layer 所对应的 global ids（block ids）：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 16cd1f8..ab1e640 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1,4 +1,4 @@
-from typing import Iterator, Optional
+from typing import Iterator, NamedTuple, Optional
 
 import torch
 from roseinfer.detokenizer import (
@@ -63,6 +63,18 @@ class InferenceEngine:
             return PrefixDiffDetokenizer(self.tokenizer)
 
         self._make_detok = make_detok
+        block_size = 64
+        max_context = max_position_embeddings or self.config.max_position_embeddings
+        max_blocks_per_layer = (max_context + block_size - 1) // block_size
+        self.kv_manager = KVBlockManager(
+            num_layers=self.config.n_layers,
+            num_heads=self.config.n_heads,
+            head_dim=self.config.d_model // self.config.n_heads,
+            block_size=block_size,
+            max_blocks_per_layer=max_blocks_per_layer,
+            device=self.device,
+            dtype=self.amp_dtype if self.use_amp else self.model.dtype,
+        )
 
         if self.config.vocab_size < self.tokenizer.vocab_size:
             raise ValueError("the model vocab_size is less than tokenizer vocab_size")
@@ -603,6 +615,11 @@ class InferenceSession:
         self.do_sample: bool = False
         self.stop_on_eos: bool = True
         self.step_count: int = 0
+        self.kv_manager = engine.kv_manager
+        self.block_ids_per_layer: list[list[int]] = [
+            [] for _ in range(self.kv_manager.num_layers)
+        ]
+        self.prompt_length: int = 0
 
     def set_generation_config(
         self,
@@ -633,6 +650,28 @@ class InferenceSession:
             skip_special_tokens=True,
         )
 
+    def _register_prefill_kv(
+        self,
+        presents,
+        seq_len: int,
+    ) -> None:
+        if self.kv_manager is None:
+            return
+        self.prompt_length = seq_len
+        self.block_ids_per_layer = [[] for _ in range(self.kv_manager.num_layers)]
+        for layer_idx, layer_past in enumerate(presents):
+            if layer_idx >= self.kv_manager.num_layers:
+                break
+            key, value = layer_past  # [B, H, T, D]
+            if key.size(2) != seq_len:
+                continue
+            block_ids = self.kv_manager.register_prefill_layer(
+                layer_idx,
+                key,
+                value,
+            )
+            self.block_ids_per_layer[layer_idx] = block_ids
+
     @torch.no_grad()
     def prefill(
         self,
@@ -659,6 +698,7 @@ class InferenceSession:
                 past_key_values=None,
                 use_cache=True,
             )
+        self._register_prefill_kv(presents, input_ids.size(1))
         self.kv_cache = presents
         return logits  # [..., T0, vocab]
 
@@ -694,6 +734,8 @@ class InferenceSession:
                 past_key_values=None,
                 use_cache=True,
             )
+        if input_ids.size(0) == 1:  # temporarily only support batch size 1
+            self._register_prefill_kv(presents, input_ids.size(1))
         self.kv_cache = presents
         last_logits = logits[:, -1, :]  # [batch, vocab]
         return last_logits
@@ -757,6 +799,15 @@ class InferenceSession:
             self.finished = True
         return token_id
 
+    def release_kv_blocks(self) -> None:
+        if self.kv_manager is None:
+            return
+        for layer_idx, block_ids in enumerate(self.block_ids_per_layer):
+            if not block_ids:
+                continue
+            self.kv_manager.free_blocks(layer_idx, block_ids)
+        self.block_ids_per_layer = [[] for _ in range(self.kv_manager.num_layers)]
+
     @torch.no_grad()
     def decode_step_batch(
         self,
@@ -863,4 +914,98 @@ class OfflineScheduler:
         outputs: dict[int, str] = {}
         for rid, session in self._sessions.items():
             outputs[rid] = session.decode_text()
+        for session in self._sessions.values():
+            session.release_kv_blocks()
         return outputs
+
+
+class KVBlockInfo(NamedTuple):
+    layer: int
+    block_index: int
+    start: int
+    length: int
+
+
+class KVBlockManager:
+    def __init__(
+        self,
+        num_layers: int,
+        num_heads: int,
+        head_dim: int,
+        block_size: int,
+        max_blocks_per_layer: int,
+        device: torch.device,
+        dtype: torch.dtype,
+    ) -> None:
+        self.num_layers = num_layers
+        self.num_heads = num_heads
+        self.head_dim = head_dim
+        self.block_size = block_size
+        self.max_blocks_per_layer = max_blocks_per_layer
+        self.device = device
+        self.dtype = dtype
+        self._next_block_index: list[int] = [0 for _ in range(num_layers)]
+        self._free_block_indices: list[list[int]] = [[] for _ in range(num_layers)]
+        self._block_infos: dict[int, KVBlockInfo] = {}
+
+    def _alloc_block_index(self, layer_idx: int) -> int:
+        free_list = self._free_block_indices[layer_idx]
+        if free_list:
+            return free_list.pop()
+        idx = self._next_block_index[layer_idx]
+        if idx >= self.max_blocks_per_layer:
+            raise RuntimeError(f"no more blocks available for layer {layer_idx}")
+        self._next_block_index[layer_idx] += 1
+        return idx
+
+    def _to_global_block_id(
+        self,
+        layer_idx: int,
+        block_index: int,
+    ) -> int:
+        return layer_idx * self.max_blocks_per_layer + block_index
+
+    def register_prefill_layer(
+        self,
+        layer_idx: int,
+        key: torch.Tensor,
+        value: torch.Tensor,
+    ) -> list[int]:
+        assert layer_idx < self.num_layers
+        seq_len = key.size(2)
+        block_size = self.block_size
+        num_blocks = (seq_len + block_size - 1) // block_size
+        block_ids: list[int] = []
+        for i in range(num_blocks):
+            start = i * block_size
+            end = min(start + block_size, seq_len)
+            length = end - start
+            block_idx = self._alloc_block_index(layer_idx)
+            global_id = self._to_global_block_id(
+                layer_idx,
+                block_idx,
+            )
+            info = KVBlockInfo(
+                layer=layer_idx,
+                block_index=block_idx,
+                start=start,
+                length=length,
+            )
+            self._block_infos[global_id] = info
+            block_ids.append(global_id)
+        return block_ids
+
+    def free_blocks(
+        self,
+        layer_idx: int,
+        block_ids: list[int],
+    ) -> None:
+        for global_id in block_ids:
+            info = self._block_infos.pop(global_id, None)
+            if info is None:
+                continue
+            if info.layer != layer_idx:
+                continue
+            self._free_block_indices[layer_idx].append(
+                info.block_index,
+            )

```



## 运行

重跑之前的脚本，确保一切正常：

```shell

(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm/rosellm$ ./generate.sh 
[roseinfer] device: cuda
[roseinfer] use_amp: True
[roseinfer] prompt: hi, 
[roseinfer] streaming output: ˈp, the early development and and the state at the two weeks to be considered the health or by a few in their way to be that are so on that you you will create the environment.
There’s all the two or the new new new business and a significant by the more to see it would look at a well have a way. It have this can do more of the use and the world-t be a little for the year by it, "Pt.
B, a way to the “The way it,”. The first the life will be the process. By the world-
The the same type of the future to have you can help.
```

