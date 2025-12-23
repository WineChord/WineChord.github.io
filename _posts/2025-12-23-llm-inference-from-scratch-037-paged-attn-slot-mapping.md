---
classes: wide2
title: "从零实现 LLM Inference：037. Paged Attention Slot Mapping"
excerpt: "block_table 不再每步 gather/copy：把它常驻 GPU，用 slot_mapping 让 triton kernel 直接索引，decode 吞吐更稳。"
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

上一版我们把 `block_table row` 按 session 缓存了，但 decode 这条路径里仍然有个绕不开的事情：

- kernel 要吃的是 `[B, MAX_BLOCKS]`
- 我们手上的是“每个 session 自己的 block_ids”

所以每一步都要把它 **materialize 成 batch view**，再把整块 `block_tables` 拷到 GPU。

这次的目标很明确：**让 `block_table` 常驻 GPU**，每个 session 占一个固定的 slot（类似 vLLM 的 `max_num_seqs` 思路），decode 每步只传一个很小的 `slot_mapping=[B]`，kernel 内部自己去 `block_table[slot]` 找那一行。

这样 `block_table` 的更新也变成“脏更新”：

- 大部分 token 只是写进当前 block，`block_ids_per_layer` 不变 → 不需要动 `block_table`
- 只有在 block 边界（每 64 token）或 COW 时，`block_ids` 变化 → 才同步那几行

## 代码变更

### `rosetrainer/paged_attention.py`

核心就是把 kernel 的 `block_table` 从“按 batch 排列”改成“按 slot 排列”，并新增 `slot_mapping`：

```diff
diff --git a/rosellm/rosetrainer/paged_attention.py b/rosellm/rosetrainer/paged_attention.py
@@
 class PagedKVCache:
@@
     block_tables: list[torch.Tensor]
+    slot_mapping: torch.Tensor
@@
-def paged_attention_decode_triton(..., block_table: torch.Tensor, context_lens: torch.Tensor, ...):
+def paged_attention_decode_triton(..., block_table: torch.Tensor, slot_mapping: torch.Tensor, context_lens: torch.Tensor, ...):
@@
 def _paged_attn_decode_kernel(..., block_table_ptr, slot_mapping_ptr, context_lens_ptr, ...):
@@
+    slot = tl.load(slot_mapping_ptr + b).to(tl.int32)
@@
-    block_id = tl.load(block_table_ptr + b * MAX_BLOCKS + lb, ...)
+    block_id = tl.load(block_table_ptr + slot * MAX_BLOCKS + lb, ...)
```

ref 路径同样按 `slot_mapping` 去索引 `block_table`（方便 CPU 校验）。

### `roseinfer/engine.py`

1. 增加 slot 管理（`_alloc_paged_slot/_free_paged_slot`）和全局 `global_block_tables=[n_layers, slot_cap, max_blocks]`。
2. decode 时如果 session 还没有 slot，就分配一个；并在 `release_kv_blocks()` 里释放 slot。
3. `sync_global_block_tables`：只在 “layer0 的 `(len(block_ids), last_id)` 变化” 时，把 dirty sessions 的 rows 同步到 GPU 的 `global_block_tables`。
4. 每步只构造 `slot_mapping=[B]`，把 `global_block_tables[layer]` 直接丢给 `PagedKVCache`（不再做 `[B, MAX_BLOCKS]` 的 gather）。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
+self._paged_global_block_tables: torch.Tensor | None = None
+self._paged_slot_capacity: int = 0
+self._paged_free_slots: list[int] = []
@@
 if self.use_paged_attention:
 @@
+    slot_ids: list[int] = []
+    for sess in sessions:
+        if sess.paged_slot_id is None:
+            sess.paged_slot_id = self._alloc_paged_slot()
+            sess.clear_paged_block_table_cache()
+        assert sess.paged_slot_id is not None
+        slot_ids.append(sess.paged_slot_id)
+    global_block_tables = self._get_paged_global_block_tables()
+
+    # dirty sync（只在 block 变化时发生）
+    dirty_idx: list[int] = []
+    for idx, sess in enumerate(sessions):
+        _, dirty = sess.get_paged_block_table_row_cpu_and_dirty(layer_idx=0, offset=0)
+        if dirty:
+            dirty_idx.append(idx)
+    if dirty_idx:
+        dirty_slot_ids = [slot_ids[idx] for idx in dirty_idx]
+        dirty_slot_ids_t = torch.tensor(dirty_slot_ids, device=device, dtype=torch.long)
+        for layer_idx in range(num_layers):
+            offset = layer_idx * max_blocks_per_layer
+            rows = [
+                sessions[idx].get_paged_block_table_row_cpu(layer_idx=layer_idx, offset=offset)
+                for idx in dirty_idx
+            ]
+            ...
+            global_block_tables[layer_idx].index_copy_(0, dirty_slot_ids_t, rows_buf)
+
+    slot_mapping = torch.tensor(slot_ids, device=device, dtype=torch.int32)
+    block_tables = [global_block_tables[layer_idx] for layer_idx in range(num_layers)]
+    paged = PagedKVCache(..., block_tables=block_tables, slot_mapping=slot_mapping, ...)
```

### `rosetrainer/model.py`

把 `slot_mapping` 透传到 `paged_attention_decode_*`：

```diff
diff --git a/rosellm/rosetrainer/model.py b/rosellm/rosetrainer/model.py
@@
                 block_table=paged_kv_cache.block_tables[layer_idx],
+                slot_mapping=paged_kv_cache.slot_mapping,
```

## 运行

```bash
pytest -q
```

```text
22 passed, 1 warning in 1.63s
```

## Benchmark（HF GPT-2）

命令保持和上一版一致（把变量尽量收敛到 decode path）：

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn
```

### Before

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 7.554661/7.554605 s
Total time p50/mean: 7.723301/7.726726 s
Throughput(completion,decode) p50/mean: 4337.45/4337.51 tokens/s
Throughput(completion,total) p50/mean: 4242.75/4240.89 tokens/s
TTFT p50/mean: 2.70/2.69 ms
TPOT p50/mean: 14.82/14.82 ms/token
Latency p50/mean: 7573.78/7574.42 ms
```

### After

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 7.433242/7.436703 s
Total time p50/mean: 7.607697/7.610673 s
Throughput(completion,decode) p50/mean: 4408.31/4406.26 tokens/s
Throughput(completion,total) p50/mean: 4307.22/4305.53 tokens/s
TTFT p50/mean: 2.70/2.70 ms
TPOT p50/mean: 14.59/14.59 ms/token
Latency p50/mean: 7457.71/7460.41 ms
```

## 结论

- decode 吞吐：`4337.51 → 4406.26 tokens/s`（+1.58%）
- TPOT：`14.82 → 14.59 ms/token`（-1.55%）
- 核心原因：`block_table` 不再每步 materialize 成 `[B, MAX_BLOCKS]`，只在 block 变化时做 dirty sync；每步只传 `slot_mapping` 给 kernel。
