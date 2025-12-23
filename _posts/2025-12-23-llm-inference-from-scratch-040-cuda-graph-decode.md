---
classes: wide2
title: "从零实现 LLM Inference：040. CUDA Graph 加速 paged decode"
excerpt: "decode 热路径里 kernel launch 太密？把一次 decode step 捕获成 CUDA Graph，replay 省掉大量 CPU dispatch。"
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

paged attention 跑起来以后，GPU 这边已经很“实”了，但 decode 依然有一个很典型的瓶颈：**kernel launch 太密**。

以 GPT-2 为例：

- `L=12` 层
- decode `T=512` 步

这意味着 **一次 run 至少有 `12*512=6144` 次 attention kernel launch**，再算上 qkv/ffn/ln 等等，launch 数量更夸张。很多时候 GPU 并不是算不动，而是 CPU 在不停地 dispatch。

CUDA Graph 的核心价值就是：**把这段固定形状的 CUDA work 录下来，后面直接 replay**，把大量 CPU-side dispatch 开销砍掉。

这版只做一件事：给 paged decode(T=1) 加一个可选的 CUDA Graph fast-path。

不使用 cuda graph 和使用 cuda graph 的 profile 对比：

![image-20251223203415698](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251223203415698.png)

## 代码变更

### `roseinfer/engine.py`

做法很直接：

1. `InferenceEngine` 增加 `use_cuda_graph` 开关
2. 维护一个 `batch_size -> graph` 的 cache（batch size 固定时收益最好）
3. graph 里 capture 的是 **一次完整的 model forward（paged decode, use_cache=True）**
4. 每个 step 只做：
   - 把新的 `input_ids/position_ids/slot_mapping/context_lens` 拷进静态 buffer
   - `graph.replay()`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
+@dataclass
+class _PagedDecodeCudaGraph:
+    batch_size: int
+    global_block_tables_ptr: int
+    graph: torch.cuda.CUDAGraph
+    input_ids: torch.Tensor
+    position_ids: torch.Tensor
+    slot_mapping: torch.Tensor
+    context_lens: torch.Tensor
+    logits: torch.Tensor
+    presents: list[tuple[torch.Tensor, torch.Tensor]]
@@
 class InferenceEngine:
     def __init__(
@@
-        use_paged_attention: bool = False,
+        use_paged_attention: bool = False,
+        use_cuda_graph: bool = False,
@@
+        self.use_cuda_graph = (
+            bool(use_cuda_graph)
+            and self.device.type == "cuda"
+            and torch.cuda.is_available()
+        )
+        self._paged_decode_cuda_graphs: dict[int, _PagedDecodeCudaGraph] = {}
+        self._cuda_graph_pool = (
+            torch.cuda.graphs.graph_pool_handle() if self.use_cuda_graph else None
+        )
@@
+    def _get_or_create_paged_decode_cuda_graph(
+        self,
+        *,
+        batch_size: int,
+        global_block_tables: torch.Tensor,
+    ) -> _PagedDecodeCudaGraph:
+        ...
+        # warmup -> torch.cuda.graph(...) capture -> cache by batch_size
+        ...
@@ def decode_step_sessions(self, sessions):
-                paged = PagedKVCache(...)
-                logits, _, presents = self.model(..., paged_kv_cache=paged)
+                if self.use_cuda_graph:
+                    graph = self._get_or_create_paged_decode_cuda_graph(...)
+                    graph.input_ids[:, 0].copy_(torch.tensor(last_ids, dtype=torch.long), non_blocking=True)
+                    graph.position_ids[:, 0].copy_(torch.tensor(seq_lens, dtype=torch.long), non_blocking=True)
+                    graph.slot_mapping.copy_(torch.tensor(slot_ids, dtype=torch.int32), non_blocking=True)
+                    graph.context_lens.copy_(torch.tensor(seq_lens, dtype=torch.int32), non_blocking=True)
+                    graph.graph.replay()
+                    logits = graph.logits
+                    presents = graph.presents
+                else:
+                    paged = PagedKVCache(...)
+                    logits, _, presents = self.model(..., paged_kv_cache=paged)
```

注意点：

- **必须 warmup 再 capture**：否则 Triton 首次编译/allocator 扩容会直接让 capture 失败
- graph 只对 **固定 batch size + decode(T=1)** 这种热路径最友好；batch size 变了就走 fallback

### `roseinfer/benchmark_scheduler.py`

加一个开关 `--cuda-graph`：

```diff
diff --git a/rosellm/roseinfer/benchmark_scheduler.py b/rosellm/roseinfer/benchmark_scheduler.py
@@
     parser.add_argument(
         "--paged-attn",
         action="store_true",
         help="Use paged attention",
     )
+    parser.add_argument(
+        "--cuda-graph",
+        action="store_true",
+        help="Use CUDA Graph for decode(T=1) when possible (CUDA only).",
+    )
@@
         engine = InferenceEngine(
@@
             use_paged_attention=args.paged_attn,
+            use_cuda_graph=args.cuda_graph,
         )
```

## 运行

```bash
pytest -q
```

```text
......................                                                   [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
22 passed, 1 warning in 1.61s
```

## Benchmark（HF GPT-2）

命令（固定 workload，收敛到 decode 热路径）：

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn
```

### Before（不开 CUDA Graph）

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 3.391189/3.392113 s
Total time p50/mean: 3.570067/3.568646 s
Throughput(completion,decode) p50/mean: 9662.69/9660.06 tokens/s
Throughput(completion,total) p50/mean: 9178.54/9182.20 tokens/s
TTFT p50/mean: 2.78/2.75 ms
TPOT p50/mean: 6.67/6.68 ms/token
Latency p50/mean: 3412.77/3413.99 ms
```

### After（开 CUDA Graph）

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn --cuda-graph
```

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 3.102104/3.101720 s
Total time p50/mean: 3.276822/3.277024 s
Throughput(completion,decode) p50/mean: 10563.15/10564.46 tokens/s
Throughput(completion,total) p50/mean: 9999.93/9999.32 tokens/s
TTFT p50/mean: 2.72/2.72 ms
TPOT p50/mean: 6.11/6.11 ms/token
Latency p50/mean: 3125.11/3124.65 ms
```

## 结论

CUDA Graph 这类优化非常“朴素”：不改算子、不改模型结构，只是把 decode 热路径的 **CPU dispatch** 切成 replay。

在这组 GPT-2 的数据上：

- decode 吞吐（mean）：`9660.06 → 10564.46 tokens/s`（约 **+9.36%**）
- TPOT（mean）：`6.68 → 6.11 ms/token`（约 **-8.53%**）

后面如果要继续往前推：

- 让更多 batch size 命中 graph cache（或者做 padding/分桶）
- 把 graph replay 的范围扩到更多 decode 侧的固定开销（比如一些固定形状的 tensor 构造/拷贝）

