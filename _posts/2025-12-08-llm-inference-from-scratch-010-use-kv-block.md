---
classes: wide2
title: "从零实现 LLM Inference：010. Use KV Block"
excerpt: "让 kv block manager 真正发挥作用，实现 python 版 paged attention。"
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

在完成了 batch decoding 之后，接下来我们可以考虑让我们的 kv block manager 真正发挥作用，而不是仅仅记录 metadata，事实上我们要开始在 python 侧实现一个简要的 paged attention，在前向过程中使用我们之前管理的 kv blocks，并且在每次 decode 之后把新生成的 kv 给更新到每一个 layer 的 block 里。

为此，我们需要在 prefill 之后，拿到 prompt_ids 对应的 kv-cache，把这个最初的 kv-cache 交给 kv block manager 进行管理，按 layer 来构建 kv block，然后在后面的每一次 decode 之前，根据每个 session 的 block id 列表，去 block manager gather 到用于 decode 的所有 kv cache，在 decode forward 完成之后，从返回的新的 kv-cache 中拿最后一个 k v pair，把这个新的 kv-cache 添加到 kv block 当中（按需扩新的 block）

## 代码变更

### `engine.py`

核心改动点就是 InferenceEngine 里面的 decode_step_sessions，这一步会在开始的时候用 kv block manager 的 gather sequence （KVBlockManager 的 gather_sequence 实现）来从 kv blocks 里面拿出来这些 sessions 做这一步 forward 时需要的 kv-cache，并在 forward 之后拿新的 kv-cache 更新到 kv block 当中（KVBlockManager 的 append_token 实现），此外就是 prefill 之后需要调用的 register_prefill_layer 的实现需要把 prefill 产生的 kv-cache 做到 kv block 当中：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 4fbf631..3c69649 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -610,29 +610,33 @@ class InferenceEngine:
 
         device = self.device
         batch_size = len(sessions)
+        kvm = self.kv_manager
+
         last_ids: list[int] = []
         seq_lens: list[int] = []
         for sess in sessions:
             if sess.finished:
                 continue
-            assert sess.kv_cache is not None
             assert sess.generated_ids
             last_ids.append(sess.generated_ids[-1])
-            key0, _ = sess.kv_cache[0]
-            seq_lens.append(key0.size(2))
+            seq_len = sess.prompt_length + sess.step_count - 1
+            seq_lens.append(seq_len)
         assert len(last_ids) == batch_size
         lens = torch.tensor(seq_lens, device=device, dtype=torch.long)
         max_len = max(seq_lens)
+
         input_ids = torch.tensor(  # [B, 1]
             last_ids,
             dtype=torch.long,
             device=device,
         ).view(batch_size, 1)
-        past_mask = torch.arange(  # [B, max_len], bool
+        past_mask = torch.arange(
             max_len,
             device=device,
-        ).unsqueeze(0) < lens.unsqueeze(1)
-        new_mask = torch.ones(  # [B, 1]
+        ).unsqueeze(
+            0
+        ) < lens.unsqueeze(1)
+        new_mask = torch.ones(
             batch_size,
             1,
             device=device,
@@ -644,42 +648,48 @@ class InferenceEngine:
         ).to(torch.long)
 
         batched_past = []
-        num_layers = len(sessions[0].kv_cache)
+        num_layers = kvm.num_layers
         for layer_idx in range(num_layers):
             k_list = []
             v_list = []
             for idx, sess in enumerate(sessions):
-                k_layer, v_layer = sess.kv_cache[layer_idx]
-                T_i = seq_lens[idx]
+                seq_len = seq_lens[idx]
+                block_ids = sess.block_ids_per_layer[layer_idx]
+                k_seq, v_seq = kvm.gather_sequence(
+                    layer_idx,
+                    block_ids,
+                    seq_len,
+                )  # [1, H, T_i, D]
+                T_i = k_seq.size(2)
                 if T_i < max_len:
                     pad_len = max_len - T_i
                     pad_shape = (
                         1,
-                        k_layer.size(1),
+                        k_seq.size(1),
                         pad_len,
-                        k_layer.size(3),
+                        k_seq.size(3),
                     )
                     k_pad = torch.zeros(
                         pad_shape,
-                        dtype=k_layer.dtype,
-                        device=k_layer.device,
+                        dtype=k_seq.dtype,
+                        device=k_seq.device,
                     )
                     v_pad = torch.zeros(
                         pad_shape,
-                        dtype=v_layer.dtype,
-                        device=v_layer.device,
+                        dtype=v_seq.dtype,
+                        device=v_seq.device,
                     )
                     k_full = torch.cat(
-                        [k_layer, k_pad],
+                        [k_seq, k_pad],
                         dim=2,
                     )
                     v_full = torch.cat(
-                        [v_layer, v_pad],
+                        [v_seq, v_pad],
                         dim=2,
                     )
                 else:
-                    k_full = k_layer
-                    v_full = v_layer
+                    k_full = k_seq
+                    v_full = v_seq
                 k_list.append(k_full)
                 v_list.append(v_full)
             k_cat = torch.cat(k_list, dim=0)
@@ -711,11 +721,24 @@ class InferenceEngine:
             for idx, sess in enumerate(sessions):
                 if sess.finished:
                     continue
-                prev_len = seq_lens[idx]
-                new_len = prev_len + 1
-                k_slice = k_b[idx : idx + 1, :, :new_len, :].contiguous()
-                v_slice = v_b[idx : idx + 1, :, :new_len, :].contiguous()
-                sess.kv_cache[layer_idx] = (k_slice, v_slice)
+                k_new = k_b[
+                    idx : idx + 1,
+                    :,
+                    max_len : max_len + 1,
+                    :,
+                ]
+                v_new = v_b[
+                    idx : idx + 1,
+                    :,
+                    max_len : max_len + 1,
+                    :,
+                ]
+                kvm.append_token(
+                    layer_idx,
+                    sess.block_ids_per_layer[layer_idx],
+                    k_new,
+                    v_new,
+                )
         return last_logits
 
 
@@ -1102,6 +1125,10 @@ class KVBlockManager:
         self._next_block_index: list[int] = [0 for _ in range(num_layers)]
         self._free_block_indices: list[list[int]] = [[] for _ in range(num_layers)]
         self._block_infos: dict[int, KVBlockInfo] = {}
+        self._block_storage: dict[
+            int,
+            tuple[torch.Tensor, torch.Tensor],
+        ] = {}  # global_id -> (key_block, value_block)
 
     def _alloc_block_index(self, layer_idx: int) -> int:
         free_list = self._free_block_indices[layer_idx]
@@ -1123,10 +1150,10 @@ class KVBlockManager:
     def register_prefill_layer(
         self,
         layer_idx: int,
-        key: torch.Tensor,
+        key: torch.Tensor,  # [1, H, D, T]
         value: torch.Tensor,
     ) -> list[int]:
-        assert layer_idx < self.num_layers
+        assert 0 <= layer_idx < self.num_layers
         seq_len = key.size(2)
         block_size = self.block_size
         num_blocks = (seq_len + block_size - 1) // block_size
@@ -1135,6 +1162,8 @@ class KVBlockManager:
             start = i * block_size
             end = min(start + block_size, seq_len)
             length = end - start
+            k_slice = key[:, :, start:end, :]
+            v_slice = value[:, :, start:end, :]
             block_idx = self._alloc_block_index(layer_idx)
             global_id = self._to_global_block_id(
                 layer_idx,
@@ -1147,6 +1176,19 @@ class KVBlockManager:
                 length=length,
             )
             self._block_infos[global_id] = info
+            k_block = torch.zeros(
+                (
+                    self.num_heads,
+                    block_size,
+                    self.head_dim,
+                ),
+                dtype=self.dtype,
+                device=self.device,
+            )
+            v_block = torch.zeros_like(k_block)
+            k_block[:, :length, :] = k_slice[0]
+            v_block[:, :length, :] = v_slice[0]
+            self._block_storage[global_id] = (k_block, v_block)
             block_ids.append(global_id)
         return block_ids
 
@@ -1164,3 +1206,120 @@ class KVBlockManager:
             self._free_block_indices[layer_idx].append(
                 info.block_index,
             )
+            self._block_storage.pop(global_id, None)
+
+    def append_token(
+        self,
+        layer_idx: int,
+        block_ids: list[int],
+        key_new: torch.Tensor,
+        value_new: torch.Tensor,
+    ) -> None:
+        assert 0 <= layer_idx < self.num_layers
+        assert key_new.size(2) == 1
+        if not block_ids:
+            block_idx = self._alloc_block_index(layer_idx)
+            global_id = self._to_global_block_id(
+                layer_idx,
+                block_idx,
+            )
+            info = KVBlockInfo(
+                layer=layer_idx,
+                block_index=block_idx,
+                start=0,
+                length=0,
+            )
+            self._block_infos[global_id] = info
+            k_block = torch.zeros(
+                (
+                    self.num_heads,
+                    self.block_size,
+                    self.head_dim,
+                ),
+                dtype=self.dtype,
+                device=self.device,
+            )
+            v_block = torch.zeros_like(k_block)
+            self._block_storage[global_id] = (k_block, v_block)
+            block_ids.append(global_id)
+        last_id = block_ids[-1]
+        info = self._block_infos[last_id]
+        k_block, v_block = self._block_storage[last_id]
+        if info.length >= self.block_size:
+            block_idx = self._alloc_block_index(layer_idx)
+            global_id = self._to_global_block_id(
+                layer_idx,
+                block_idx,
+            )
+            info = KVBlockInfo(
+                layer=layer_idx,
+                block_index=block_idx,
+                start=info.start + info.length,
+                length=0,
+            )
+            self._block_infos[global_id] = info
+            k_block = torch.zeros(
+                (
+                    self.num_heads,
+                    self.block_size,
+                    self.head_dim,
+                ),
+                dtype=self.dtype,
+                device=self.device,
+            )
+            v_block = torch.zeros_like(k_block)
+            self._block_storage[global_id] = (k_block, v_block)
+            block_ids.append(global_id)
+            last_id = global_id
+        info = self._block_infos[last_id]
+        k_block, v_block = self._block_storage[last_id]
+        pos = info.length
+        k_block[:, pos, :] = key_new[0, :, 0, :]
+        v_block[:, pos, :] = value_new[0, :, 0, :]
+        new_info = KVBlockInfo(
+            layer=info.layer,
+            block_index=info.block_index,
+            start=info.start,
+            length=info.length + 1,
+        )
+        self._block_infos[last_id] = new_info
+
+    def gather_sequence(
+        self,
+        layer_idx: int,
+        block_ids: list[int],
+        total_len: int,
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        assert 0 <= layer_idx < self.num_layers
+        k_seq = torch.zeros(
+            (
+                1,
+                self.num_heads,
+                total_len,
+                self.head_dim,
+            ),
+            dtype=self.dtype,
+            device=self.device,
+        )
+        v_seq = torch.zeros_like(k_seq)
+        cur = 0
+        for global_id in block_ids:
+            info = self._block_infos[global_id]
+            if info is None:
+                continue
+            if info.layer != layer_idx:
+                continue
+            k_block, v_block = self._block_storage[global_id]
+            length = info.length
+            if length <= 0:
+                continue
+            end = min(cur + length, total_len)
+            take = end - cur
+            if take <= 0:
+                break
+            k_seq[0, :, cur:end, :] = k_block[:, :take, :]
+            v_seq[0, :, cur:end, :] = v_block[:, :take, :]
+            cur = end
+            if cur >= total_len:
+                break
+        return k_seq, v_seq

```



## 运行

再执行一下之前的脚本看看：

```shell
$ ./offline_example.sh 
### request 0
hi, __________ and Tricro, __________ and to be a part of the class.
Hendhem in his most recent book of science (in terms of the contents of the book) and with an appeal to this story, Ovonz, is a fictional classic, and is fiction writer in the book of Theres. His book has the power to do with some of his works, to illustrate the power of our art and literature. He has, however, been a member of

### request 1
hello, it's been very common to live in the summer and summer. However, this is the first time in science as the scientific scientific methods and methods to guide the theory of microorganisms and their applications.
Oneel is one of the most interesting tools to look at. In the last decade, the so-called bio-harvested bio-assessment techniques are moving over three billion years ago. So, as this last study revealed that the bio-assessment technique provides the method for

### request 2
how of this is not as simple as you’ll have to use, and you’ll have to follow the puzzle.
- Consider this activity!
- Have you memorize the ideas in the picture and add a picture.
- Ask your students to draw a picture and explain how they’ve read them.
- Give the students read the picture and ask them to let them write them.
- Add the lesson to the students and ask them to write.
-
```

