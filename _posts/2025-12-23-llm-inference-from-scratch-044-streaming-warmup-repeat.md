---
classes: wide2
title: "从零实现 LLM Inference：044. Streaming Benchmark Warmup + Repeat（把 cold-start 拆出去）"
excerpt: "paged-attn / CUDA Graph 的第一次请求会把 Triton JIT 和 graph capture 算进 TTFT/吞吐，导致对比结论失真。给 benchmark_streaming 加 warmup-runs + repeat-runs，把冷启动和稳态拆开。"
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

上一版我们在 streaming benchmark 里加了 `--paged-attn/--cuda-graph/--nvtx`，很快就碰到一个“看起来很反常”的现象：

- 开 paged-attn + CUDA Graph 后，**ITL/TPOT/吞吐明显变好**
- 但 **TTFT p99 反而变得很大**（动不动 800ms+）

这不是系统真的变慢了，而是 **第一次跑** 会把：

- Triton kernel 的 JIT/compile（以及可能的 autotune）
- CUDA Graph 的 capture（第一次捕获某个 batch size）

这些一次性成本全都吃进 TTFT/吞吐里。这样用 streaming benchmark 做前后对比，结论会被“冷启动噪声”污染。

这一版的目标很简单：给 `benchmark_streaming.py` 增加 `warmup-runs + repeat-runs`，把 cold-start 和 steady-state 拆开。

## 代码变更

### `roseinfer/benchmark_streaming.py`

新增两个参数：

- `--warmup-runs N`：先跑 N 次 warmup（不输出统计），用来触发 JIT / graph capture
- `--repeat-runs M`：再跑 M 次测量，输出每次的 TTFT/TPOT/ITL/Latency p99

并把 “跑一次 streaming benchmark” 抽成 `run_once()`，main 里按 warmup/measure 两段 loop 调用。

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
     parser.add_argument("--num-requests", type=int, default=16)
+    parser.add_argument("--warmup-runs", type=int, default=0)
+    parser.add_argument("--repeat-runs", type=int, default=1)
@@
+def run_once(
+    *,
+    engine: InferenceEngine,
+    prompts: list[str],
+    prompt_lens: list[int],
+    args: argparse.Namespace,
+    record_token_timestamps: bool,
+    print_summary: bool,
+):
+    mgr = SchedulerManager(..., record_token_timestamps=record_token_timestamps)
+    ...
+    if not print_summary:
+        return
+    print("=== streaming benchmark ===")
+    ...
@@
 def main() -> None:
     args = parse_args()
+    if args.warmup_runs < 0:
+        raise ValueError("--warmup-runs must be >= 0")
+    if args.repeat_runs <= 0:
+        raise ValueError("--repeat-runs must be >= 1")
@@
+    for i in range(int(args.warmup_runs)):
+        print(f"=== warmup {i + 1}/{int(args.warmup_runs)} ===")
+        run_once(..., record_token_timestamps=False, print_summary=False)
+
+    for i in range(int(args.repeat_runs)):
+        if int(args.repeat_runs) > 1:
+            print(f"=== run {i + 1}/{int(args.repeat_runs)} ===")
+        run_once(..., record_token_timestamps=True, print_summary=True)
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
23 passed, 1 warning in 2.11s
```

## Benchmark（HF GPT-2 / streaming）

配置（decode-heavy，强行让 decode 走快路径）：

- `B=64`：`--num-requests 64 --max-batch-size 64`
- `T=256`：`--max-new-tokens 256`
- submit interval = 0：一次性灌满

命令：

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --unique-prompts \
  --num-requests 64 --max-new-tokens 256 --max-batch-size 64 \
  --submit-interval-ms 0 --no-stop-on-eos --no-prefix-cache \
  --paged-attn --cuda-graph
```

### Before（无 warmup）

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
Submit wall: 0.084546 s
add_request latency p50/p95/p99: 0.03/0.05/28.68 ms
TTFT p50/p95/p99: 880.48/883.14/883.77 ms
TPOT p50/p95/p99: 5.68/5.68/6.61 ms/token
ITL p50/p95/p99: 4.83/5.81/16.12 ms
Latency p50/p95/p99: 2332.72/2334.55/2335.08 ms
Throughput (completion,total): 6782.33 tokens/s
```

### After（warmup + repeat=3）

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --unique-prompts \
  --num-requests 64 --max-new-tokens 256 --max-batch-size 64 \
  --submit-interval-ms 0 --no-stop-on-eos --no-prefix-cache \
  --paged-attn --cuda-graph \
  --warmup-runs 1 --repeat-runs 3
```

```text
=== warmup 1/1 ===
=== run 1/3 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 64
Prompt tokens (total): 256
Completion tokens (total): 16384
Submit wall: 0.006119 s
add_request latency p50/p95/p99: 0.04/0.06/0.11 ms
TTFT p50/p95/p99: 31.81/34.16/34.44 ms
TPOT p50/p95/p99: 5.20/5.20/5.23 ms/token
ITL p50/p95/p99: 4.94/5.47/15.84 ms
Latency p50/p95/p99: 1361.67/1362.68/1363.08 ms
Throughput (completion,total): 11991.59 tokens/s
=== run 2/3 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 64
Prompt tokens (total): 256
Completion tokens (total): 16384
Submit wall: 0.006467 s
add_request latency p50/p95/p99: 0.04/0.07/0.15 ms
TTFT p50/p95/p99: 31.40/34.19/34.44 ms
TPOT p50/p95/p99: 5.12/5.12/5.15 ms/token
ITL p50/p95/p99: 4.85/5.54/15.65 ms
Latency p50/p95/p99: 1337.98/1340.40/1341.19 ms
Throughput (completion,total): 12196.65 tokens/s
=== run 3/3 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 64
Prompt tokens (total): 256
Completion tokens (total): 16384
Submit wall: 0.006207 s
add_request latency p50/p95/p99: 0.04/0.08/0.11 ms
TTFT p50/p95/p99: 31.29/33.73/34.06 ms
TPOT p50/p95/p99: 5.30/5.30/5.33 ms/token
ITL p50/p95/p99: 5.05/5.83/15.36 ms
Latency p50/p95/p99: 1385.70/1386.90/1387.26 ms
Throughput (completion,total): 11783.53 tokens/s
```

## 结论

这版没有改任何推理逻辑，只是在 benchmark 里把“冷启动一次性成本”拆出去，得到更可信的对比口径：

- TTFT p99：`883.77ms -> ~34ms`（约 `26x`，本质是把 Triton JIT / graph capture 从 TTFT 里移走）
- Throughput：`6782 -> ~12000 tokens/s`（约 `1.7x`，冷启动被吞掉的时间不再算进来）
- repeat=3 的三个 run 很接近：说明这是稳态数据，适合拿来做优化前后的量化对比

后面凡是要对比 paged-attn / CUDA Graph / Triton kernel 的收益，我会默认用：

```bash
--warmup-runs 1 --repeat-runs 3
```

