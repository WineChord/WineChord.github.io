---
classes: wide2
title: "从零实现 LLM Inference：053. CUDA Graph Host Buffers（把 metadata copy 变成真·non_blocking）"
excerpt: "paged-attn + CUDA Graph decode 里，每步都在 torch.tensor(list) 然后 copy_ 到 GPU，既有分配也不是真正 non_blocking。改成复用 pinned host buffer + numpy view 直接写入，吞吐提升、TPOT 下降。"
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

前几版我们一直在抠 streaming / scheduler 的 CPU overhead，但只要你把 decode 跑快（比如 `--paged-attn --cuda-graph`），很多“看起来无关紧要”的小开销就会被放大。

最典型的就是这段逻辑（paged-attn + cuda graph replay 前）：

- 每步把 `last_ids/seq_lens/slot_ids` 这些 **Python list** 转成 `torch.tensor(...)`
- 再 `copy_` 到 graph 的输入 buffer
- 还会写 `non_blocking=True`，但 **source 不是 pinned memory 的话这并不是真的 non_blocking**（同时每步还有分配/释放）

这一版目标很明确：**把这些 metadata 的 H2D copy 变成真正的 pinned + non_blocking，并且把每步的 tensor 分配去掉。**

## 代码变更

### `roseinfer/engine.py`

核心思路：

1) 在 `_PagedDecodeCudaGraph` 里新增一组 **CPU pinned staging buffer**（以及对应的 numpy view）
2) 每步 decode 时，直接用 numpy 把 list 写进 pinned buffer（非常快）
3) 再从 pinned buffer `copy_` 到 graph 的 GPU buffer（这才是有效的 `non_blocking=True`）

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
+import numpy as np
@@
 class _PagedDecodeCudaGraph:
@@
+    input_ids_host: torch.Tensor  # [B] int64 cpu (pinned)
+    position_ids_host: torch.Tensor  # [B] int64 cpu (pinned)
+    slot_mapping_host: torch.Tensor  # [B] int32 cpu (pinned)
+    context_lens_host: torch.Tensor  # [B] int32 cpu (pinned)
+    input_ids_host_np: np.ndarray
+    position_ids_host_np: np.ndarray
+    slot_mapping_host_np: np.ndarray
+    context_lens_host_np: np.ndarray
@@
 def _get_or_create_paged_decode_cuda_graph(...):
@@
+    input_ids_host = torch.empty((B,), device="cpu", dtype=torch.long, pin_memory=True)
+    position_ids_host = torch.empty((B,), device="cpu", dtype=torch.long, pin_memory=True)
+    slot_mapping_host = torch.empty((B,), device="cpu", dtype=torch.int32, pin_memory=True)
+    context_lens_host = torch.empty((B,), device="cpu", dtype=torch.int32, pin_memory=True)
@@
     captured = _PagedDecodeCudaGraph(
@@
+        input_ids_host=input_ids_host,
+        position_ids_host=position_ids_host,
+        slot_mapping_host=slot_mapping_host,
+        context_lens_host=context_lens_host,
+        input_ids_host_np=input_ids_host.numpy(),
+        position_ids_host_np=position_ids_host.numpy(),
+        slot_mapping_host_np=slot_mapping_host.numpy(),
+        context_lens_host_np=context_lens_host.numpy(),
     )
@@
 if self.use_cuda_graph:
     graph = self._get_or_create_paged_decode_cuda_graph(...)
-    graph.input_ids[:,0].copy_(torch.tensor(last_ids), non_blocking=True)
-    graph.position_ids[:,0].copy_(torch.tensor(seq_lens), non_blocking=True)
-    graph.slot_mapping.copy_(torch.tensor(slot_ids), non_blocking=True)
-    graph.context_lens.copy_(torch.tensor(seq_lens), non_blocking=True)
+    graph.input_ids_host_np[:] = last_ids
+    graph.position_ids_host_np[:] = seq_lens
+    graph.slot_mapping_host_np[:] = slot_ids
+    graph.context_lens_host_np[:] = seq_lens
+    graph.input_ids[:,0].copy_(graph.input_ids_host, non_blocking=True)
+    graph.position_ids[:,0].copy_(graph.position_ids_host, non_blocking=True)
+    graph.slot_mapping.copy_(graph.slot_mapping_host, non_blocking=True)
+    graph.context_lens.copy_(graph.context_lens_host, non_blocking=True)
```

## 运行

```bash
pytest -q
```

```text
..............................                                           [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 1 warning in 2.12s
```

## Benchmark（HF GPT-2 / streaming / paged-attn + cuda graph）

这一组刻意打开 `--paged-attn --cuda-graph`，让 decode 更快、CPU side 的 jitter 更显眼。

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --pretok --tokenize-workers 0 \
  --stream-interval 8 \
  --num-requests 64 \
  --warmup-runs 1 --repeat-runs 1 \
  --submit-interval-ms 20 --submit-schedule absolute \
  --max-batch-size 64 --prefill-max-batch-size 64 \
  --max-new-tokens 128 \
  --no-stop-on-eos --no-prefix-cache \
  --paged-attn --cuda-graph
```

### Before

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Tokenize workers: 0
Stream interval: 8
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 1.308910 s
Submit interval/schedule: 20.000 ms / absolute
Submit lag p50/p95/p99: 18.33/54.12/57.65 ms
add_request latency p50/p95/p99: 0.02/0.08/0.09 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 7.45/78.36/80.96 ms
Prefill->first token p50/p95/p99: 4.75/5.04/7.04 ms
TTFT p50/p95/p99: 12.33/83.08/85.95 ms
TPOT p50/p95/p99: 12.51/13.29/13.87 ms/token
ITL p50/p95/p99: 4.09/77.69/85.01 ms
Latency p50/p95/p99: 1602.08/1699.43/1770.23 ms
Throughput (completion,total): 3042.19 tokens/s
```

### After（host buffers）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Tokenize workers: 0
Stream interval: 8
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 1.296431 s
Submit interval/schedule: 20.000 ms / absolute
Submit lag p50/p95/p99: 23.90/55.74/57.10 ms
add_request latency p50/p95/p99: 0.02/0.07/0.08 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 7.43/77.83/78.57 ms
Prefill->first token p50/p95/p99: 4.70/5.01/5.90 ms
TTFT p50/p95/p99: 12.21/82.61/83.27 ms
TPOT p50/p95/p99: 11.77/13.08/13.63 ms/token
ITL p50/p95/p99: 4.02/78.46/80.44 ms
Latency p50/p95/p99: 1509.76/1672.10/1738.46 ms
Throughput (completion,total): 3210.51 tokens/s
```

## 结论

这版本质是把 “每步 decode 的 metadata 准备” 从：

- `torch.tensor(list)`（分配 + 拷贝）
- pageable host → device（`non_blocking` 名义上开了但不一定真异步）

变成：

- 写入 pinned host buffer（numpy view 写 list，几乎没成本）
- pinned host → device（这才是有效的 `non_blocking=True`）

在这组配置下能看到比较稳定的收益：

- `Throughput`: 3042.19 → 3210.51 tokens/s（+5.5%）
- `TPOT p50`: 12.51 → 11.77 ms/token（-5.9%）
- `Latency p50`: 1602.08 → 1509.76 ms（-5.8%）

后面如果继续往上抠：

- `dirty_slot_ids_t` / dirty row 的索引更新也可以做同样的 buffer 复用
- 把 submit lag / queue wait 的长尾（p95/p99）再单独抓 trace 去定位是不是还有某个 CPU/GPU 同步点在制造抖动

