---
classes: wide2
title: "从零实现 LLM Inference：036. Paged Attention Block Table Cache"
excerpt: "paged attention decode 每步重建 block_table 太浪费：按 session 缓存 row，只在 block 变化时更新，decode 吞吐和 TPOT 更稳。"
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

上一版（Paged Attention Fast Path）把 paged decode 里不必要的 `attention_mask` 构建挪走了，同时把 `block_tables` 的 H2D copy 合并成一次 async。

但 decode 压起来后（更大 batch、更长 decode），`build_block_tables` 这段仍然会在每一步做很多重复工作：**每一层、每一轮都在从 Python list 重新构建 `rows`**。

而事实上，`block_table` 的语义是“逻辑块 -> 物理块”的映射，它不会每一步都变：

- 大部分 token 只是写进当前 block（`KVBlockInfo.length += 1`），`block_ids_per_layer[layer]` 不变
- 只有在 **block 边界**（每 64 token）或 **COW**（prefix cache 共享最后一块）时才会变化

所以这次 mini PR 做一件事：**按 session 缓存每一层的 block_table row**，只有 `block_ids` 发生变化时才重算 row；每一步只做一次 `torch.stack(..., out=...)` 把这一批 session 的 rows 拼成 `[B, max_blocks]`。

另外，为了方便用 Nsight Systems 看 timeline，我也在关键路径上加了可选 NVTX range（默认关闭）。

## 代码变更

### `roseinfer/engine.py`

1. 新增 `_maybe_nvtx_range`（由 `ROSEINFER_NVTX=1` 开关控制）。
2. `InferenceSession` 增加 `get_paged_block_table_row_cpu()`：
   - cache 维度：`[num_layers][max_blocks_per_seq]`，dtype=int32，CPU
   - signature：`(len(block_ids), last_block_id)`，变化时才重算 row
3. `build_block_tables` 不再从 Python list 生成 `rows`，改成收集 row tensor，然后 `torch.stack(..., out=block_tables_cpu[layer, :B])`。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 1479d17..e7454a6 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1,4 +1,6 @@
+import os
 from collections import OrderedDict, deque
+from contextlib import contextmanager
@@
+@contextmanager
+def _maybe_nvtx_range(name: str, enabled: bool) -> Iterator[None]:
+    if enabled:
+        torch.cuda.nvtx.range_push(name)
+        try:
+            yield
+        finally:
+            torch.cuda.nvtx.range_pop()
+    else:
+        yield
@@
             if self.use_paged_attention:
                 from rosellm.rosetrainer.paged_attention import PagedKVCache

+                nvtx = device.type == "cuda" and os.environ.get("ROSEINFER_NVTX") == "1"
                 ...
-                with record_function("roseinfer.decode_step_sessions.build_block_tables"):
+                with _maybe_nvtx_range(
+                    "roseinfer.decode_step_sessions.build_block_tables",
+                    nvtx,
+                ), record_function("roseinfer.decode_step_sessions.build_block_tables"):
                     for layer_idx in range(num_layers):
                         offset = layer_idx * max_blocks_per_layer
-                        rows: list[list[int]] = []
-                        for idx, sess in enumerate(sessions):
-                            ids = sess.block_ids_per_layer[layer_idx]
-                            ...
-                            rows.append(physical)
-                        tmp = torch.tensor(rows, dtype=torch.int32)
-                        block_tables_cpu[layer_idx, :batch_size].copy_(tmp)
+                        rows = [
+                            sess.get_paged_block_table_row_cpu(
+                                layer_idx=layer_idx,
+                                offset=offset,
+                            )
+                            for sess in sessions
+                        ]
+                        torch.stack(
+                            rows,
+                            dim=0,
+                            out=block_tables_cpu[layer_idx, :batch_size],
+                        )
@@
-                with record_function("roseinfer.decode_step_sessions.h2d_block_tables"):
+                with _maybe_nvtx_range(
+                    "roseinfer.decode_step_sessions.h2d_block_tables",
+                    nvtx,
+                ), record_function("roseinfer.decode_step_sessions.h2d_block_tables"):
                     block_tables_buf[:, :batch_size].copy_(
                         block_tables_cpu[:, :batch_size],
                         non_blocking=True,
                     )
@@
-                with record_function("roseinfer.model.forward"):
+                with _maybe_nvtx_range("roseinfer.model.forward", nvtx), record_function(
+                    "roseinfer.model.forward",
+                ):
                     ...
-                with record_function("roseinfer.kv.append_token"):
+                with _maybe_nvtx_range(
+                    "roseinfer.kv.append_token", nvtx
+                ), record_function("roseinfer.kv.append_token"):
                     ...
@@
 class InferenceSession:
@@
+        self._paged_block_table_rows_cpu: list[torch.Tensor] | None = None
+        self._paged_block_table_sig: list[tuple[int, int]] | None = None
+
+    def get_paged_block_table_row_cpu(
+        self,
+        *,
+        layer_idx: int,
+        offset: int,
+    ) -> torch.Tensor:
+        max_blocks = int(self.engine.max_blocks_per_seq)
+        if self._paged_block_table_rows_cpu is None:
+            num_layers = int(self.kv_manager.num_layers)
+            self._paged_block_table_rows_cpu = [
+                torch.zeros((max_blocks,), dtype=torch.int32, device="cpu")
+                for _ in range(num_layers)
+            ]
+            self._paged_block_table_sig = [(-1, -1) for _ in range(num_layers)]
+        assert self._paged_block_table_sig is not None
+
+        ids = self.block_ids_per_layer[layer_idx]
+        sig = (len(ids), int(ids[-1]) if ids else -1)
+        if sig != self._paged_block_table_sig[layer_idx]:
+            row = self._paged_block_table_rows_cpu[layer_idx]
+            row.zero_()
+            if ids:
+                n = min(len(ids), max_blocks)
+                row[:n].copy_(
+                    torch.tensor(
+                        [gid - offset for gid in ids[:n]],
+                        dtype=torch.int32,
+                    )
+                )
+            self._paged_block_table_sig[layer_idx] = sig
+        return self._paged_block_table_rows_cpu[layer_idx]
```

### `.gitignore`

Nsight Systems 的 report 文件比较大，顺手加了 ignore：

```diff
diff --git a/.gitignore b/.gitignore
@@
+*.nsys-rep
```

## 运行

```bash
pytest -q
```

```text
22 passed, 1 warning in 1.62s
```

## Benchmark（HF GPT-2）

为了把变量尽量收敛到 decode path：

- `--pretok`：把 tokenizer 的耗时移出计时区间
- `--no-stop-on-eos`：固定每个 request decode token 数（更可比）
- `--no-prefix-cache`：避免 prefix cache / COW 干扰
- `--max-batch-size 64`：把 decode batch 压到比较大，让 block_table overhead 更明显

### Before

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn
```

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 7.815151/7.817639 s
Total time p50/mean: 7.984901/7.988579 s
Throughput(completion,decode) p50/mean: 4192.88/4191.55 tokens/s
Throughput(completion,total) p50/mean: 4103.75/4101.86 tokens/s
TTFT p50/mean: 2.64/2.64 ms
TPOT p50/mean: 15.32/15.33 ms/token
Latency p50/mean: 7831.53/7834.13 ms
```

### After

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn
```

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 7.633327/7.628122 s
Total time p50/mean: 7.806813/7.802436 s
Throughput(completion,decode) p50/mean: 4292.75/4295.69 tokens/s
Throughput(completion,total) p50/mean: 4197.36/4199.72 tokens/s
TTFT p50/mean: 2.69/2.71 ms
TPOT p50/mean: 14.98/14.97 ms/token
Latency p50/mean: 7655.36/7650.90 ms
```

## NVTX（可选）

默认不打 NVTX。需要用 nsys 看时间线时，在命令前加：

```bash
sudo -E nsys profile -o out --trace=cuda,nvtx,osrt --wait=all \
   env ROSEINFER_NVTX=1 python -m rosellm.roseinfer.benchmark_scheduler \
   --hf-model-id gpt2 --device cuda \
   --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
   --mode online --max-batch-size 64 \
   --no-stop-on-eos --no-prefix-cache --pretok \
   --warmup-runs 0 --repeat-runs 1 \
   --paged-attn
```

```bash
sudo -E ncu -o paged_attn \
  --target-processes all \
  --set full \
  --kernel-name "_paged_attn_decode_kernel" \
  --launch-count 1 \
  env ROSEINFER_NVTX=1 \
  /data/projects/rosellm/.conda/bin/python \
  -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 0 --repeat-runs 1 \
  --paged-attn

```
