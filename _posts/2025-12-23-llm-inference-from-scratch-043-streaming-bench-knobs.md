---
classes: wide2
title: "从零实现 LLM Inference：043. Streaming Benchmark 开关（paged-attn / CUDA Graph / NVTX）"
excerpt: "streaming 场景想看 TTFT/TPOT/ITL 的 p99，但之前不好一条命令切换 decode 的快路径，也不好给 nsys 挂 NVTX。补齐 benchmark 的三个开关。"
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

最近几版我一直在抠 decode 的热路径（paged-attn / CUDA Graph / Triton fuse），但如果想在 **server/streaming** 的形态下稳定对比，就需要一个“可控的、能直接切换路径”的压测入口。

`benchmark_streaming.py` 本来就会输出 TTFT/TPOT/ITL/Latency 的 p99，很适合做延迟分布对比；但它之前缺三个最关键的开关：

- `--paged-attn`：明确让 decode(T=1) 走 paged attention
- `--cuda-graph`：明确让 decode(T=1) 尝试走 CUDA Graph
- `--nvtx`：一键开启 NVTX ranges（给 nsys 看 timeline）

这一版就专注把这三个开关补齐，并且把开关状态打印出来，保证每份 benchmark 输出都是“自解释”的。

## 代码变更

### `roseinfer/benchmark_streaming.py`

- 新增 `--paged-attn/--cuda-graph/--nvtx`
- `--nvtx` 在 CUDA 下会设置 `ROSEINFER_NVTX=1`（engine 内部会用它决定是否打 NVTX）
- 创建 `InferenceEngine(...)` 时把两个开关显式传进去
- 输出里打印 `Paged attention/CUDA Graph/NVTX` 三行，方便复制对比

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
+import os
@@
+    parser.add_argument("--paged-attn", action="store_true")
+    parser.add_argument("--cuda-graph", action="store_true")
+    parser.add_argument("--nvtx", action="store_true")
@@
 def main() -> None:
     args = parse_args()
+    if args.nvtx and args.device == "cuda":
+        os.environ["ROSEINFER_NVTX"] = "1"
+    if args.cuda_graph and not args.paged_attn:
+        print("[warn] --cuda-graph is most effective with --paged-attn (decode(T=1))")
@@
     engine = InferenceEngine(
@@
+        use_paged_attention=bool(args.paged_attn),
+        use_cuda_graph=bool(args.cuda_graph),
     )
@@
         print(f"Model: {args.hf_model_id}")
         print(f"Device: {args.device}")
+        print(f"Paged attention: {bool(args.paged_attn)}")
+        print(f"CUDA Graph: {bool(args.cuda_graph)}")
+        print(f"NVTX: {bool(args.nvtx)}")
         print(f"Requests: {len(results)}")
```

## 运行

```bash
pytest -q
```

```text
.......................                                                  [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
23 passed, 1 warning in 2.08s
```

## Benchmark（HF GPT-2 / streaming）

指标含义（这版我只关心 p99）：

- **TTFT**：time to first token（收到请求 → 首 token 出来）
- **ITL**：inter-token latency（token-to-token 的间隔，越能反映 decode 的稳态）
- **TPOT**：time per output token（每个输出 token 平均耗时）
- **Latency**：单请求端到端完成时间

命令（decode-heavy，`B=64, T=256`，submit interval 设成 0）：

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --unique-prompts \
  --num-requests 64 --max-new-tokens 256 --max-batch-size 64 \
  --submit-interval-ms 0 \
  --no-stop-on-eos --no-prefix-cache
```

### Before（paged-attn=off, cuda-graph=off）

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 256
Completion tokens (total): 16384
Submit wall: 0.095108 s
add_request latency p50/p95/p99: 0.03/0.04/32.78 ms
TTFT p50/p95/p99: 205.88/208.39/222.93 ms
TPOT p50/p95/p99: 40.07/40.07/40.07 ms/token
ITL p50/p95/p99: 44.49/59.85/69.38 ms
Latency p50/p95/p99: 10425.28/10426.83/10433.11 ms
Throughput (completion,total): 1557.61 tokens/s
```

### After（paged-attn=on, cuda-graph=on）

```bash
... --paged-attn --cuda-graph
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 64
Prompt tokens (total): 256
Completion tokens (total): 16384
Submit wall: 0.085549 s
add_request latency p50/p95/p99: 0.03/0.05/29.18 ms
TTFT p50/p95/p99: 879.17/881.81/882.45 ms
TPOT p50/p95/p99: 5.85/5.85/6.78 ms/token
ITL p50/p95/p99: 5.01/5.94/16.91 ms
Latency p50/p95/p99: 2375.89/2377.21/2377.53 ms
Throughput (completion,total): 6660.03 tokens/s
```

### 对照（paged-attn=on, cuda-graph=off）

TTFT 还是很高，说明这次 TTFT 的变化主要来自 paged-attn 的 cold-start（而不是 CUDA Graph 本身）：

```bash
... --paged-attn
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 256
Completion tokens (total): 16384
Submit wall: 0.085685 s
add_request latency p50/p95/p99: 0.03/0.05/29.08 ms
TTFT p50/p95/p99: 812.87/816.17/816.76 ms
TPOT p50/p95/p99: 5.95/5.95/6.88 ms/token
ITL p50/p95/p99: 5.67/6.38/16.43 ms
Latency p50/p95/p99: 2331.57/2335.56/2361.90 ms
Throughput (completion,total): 6779.80 tokens/s
```

## 结论

这版本质是“把 streaming bench 补齐成一个可控的对比工具”，但数据也顺手说明了几个关键点：

- **稳态收益非常明显**：TPOT p99 `40.07 -> 6.78 ms/token`（约 `5.9x`），吞吐 `1557.61 -> 6660.03 tokens/s`（约 `4.3x`），Latency p99 `10433 -> 2377 ms`（约 `4.4x`）。
- **TTFT 变差是预期现象**：paged-attn 的 Triton kernel 首次触发会有 JIT/（可能还带 autotune）的 cold-start，这种开销会被 TTFT 直接吃进去。
- **下一步更合理的做法**：在 benchmark 里加 warmup，把“cold-start TTFT”和“稳态 TTFT/ITL/TPOT”拆开统计；否则 TTFT 会把 kernel 编译时间一起算进去，结论会被污染。

