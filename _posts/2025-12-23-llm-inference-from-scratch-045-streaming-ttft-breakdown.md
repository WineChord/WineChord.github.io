---
classes: wide2
title: "从零实现 LLM Inference：045. Streaming TTFT 拆解（Queue wait vs Prefill）"
excerpt: "TTFT 只是一个总数：里面既有排队等待，也有 prefill 的真实计算。只看 TTFT 很容易把优化方向搞反。给 SchedulerManager 记录 admit timestamp，并在 benchmark_streaming 里打印 TTFT breakdown。"
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

前面几版我们已经把 streaming benchmark 的“口径”做得更靠谱了（warmup/repeat + p99），但还有一个非常致命的问题：

> **TTFT 只是一个总数。**
>
> 你只看到 TTFT 变大/变小，但不知道里面到底是：
>
> - **排队等 admission**（请求还在 pending queue）
> - 还是 **prefill 真在算**（GPU/CPU 在跑算子）

在做调度（prefill admission policy、token budget、max active requests、decode-first）的时候，如果没有拆解，优化很容易走偏：你以为是算子慢，实际上是排队；你以为是排队，实际上是 prefill 太重。

这一版就做一件事：**把 TTFT 拆成两段**，并在 `benchmark_streaming` 里按 p99 打出来。

- `Queue wait`：`submit_end -> admit_ts`（从请求提交完成，到被 admission 开始 prefill 的等待）
- `Prefill->first token`：`admit_ts -> first_token_ts`（从开始 prefill，到首 token 出来）

## 代码变更

### `roseinfer/server.py`

在 `SchedulerManager` 里增加一个（仅在 `record_token_timestamps=True` 时启用的）`admit_ts` 记录：

- 每个 request 第一次进入 `scheduler.add_requests(batch)` 之前记录 `time.perf_counter()`
- 提供 `pop_admit_timestamp()` 给 benchmark 读出

核心 diff：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
 class SchedulerManager:
@@
         self._record_token_timestamps = bool(record_token_timestamps)
         self._token_timestamps: Dict[int, list[float]] = {}
+        self._admit_timestamps: Dict[int, float] = {}
@@
+    def pop_admit_timestamp(self, request_id: int) -> float | None:
+        with self._lock:
+            out = self._admit_timestamps.pop(request_id, None)
+        return float(out) if out is not None else None
@@
-                rids = self.scheduler.add_requests(batch) if batch else []
+                admit_ts = time.perf_counter() if (batch and self._record_token_timestamps) else 0.0
+                rids = self.scheduler.add_requests(batch) if batch else []
+                if rids and self._record_token_timestamps:
+                    with self._lock:
+                        for rid in rids:
+                            self._admit_timestamps[rid] = admit_ts
```

### `roseinfer/benchmark_streaming.py`

`StreamResult` 增加 `admit_ts` 字段，并新增两行输出：

- `Queue wait p50/p95/p99`
- `Prefill->first token p50/p95/p99`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
 class StreamResult:
@@
     submit_start: float
     submit_end: float
+    admit_ts: float
@@
         finish_ts = time.perf_counter()
+        admit_ts = mgr.pop_admit_timestamp(request_id) or submit_end
         token_ts = mgr.pop_token_timestamps(request_id)
@@
         add_lats = [r.submit_end - r.submit_start for r in results]
+        queue_waits = [max(0.0, r.admit_ts - r.submit_end) for r in results]
+        prefill_first = [max(0.0, r.first_token_ts - r.admit_ts) for r in results]
@@
         print(f"add_request latency ...")
+        print(f"Queue wait p50/p95/p99: ...")
+        print(f"Prefill->first token p50/p95/p99: ...")
         print(f"TTFT p50/p95/p99: ...")
```

## 运行

```bash
pytest -q
```

```text
........................                                                 [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
24 passed, 1 warning in 2.12s
```

## Benchmark（HF GPT-2 / streaming）

为了让 `Queue wait` 更明显，我用 “多请求 + 限制每轮 prefill admission” 的配置：

- `num_requests=256`
- `prefill_max_batch_size=16`（每轮最多只 admit 16 个 request 进 prefill）
- decode 走 paged-attn + CUDA Graph（保持 decode 稳态快）

命令：

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --unique-prompts \
  --num-requests 256 --max-new-tokens 64 \
  --max-batch-size 64 --prefill-max-batch-size 16 \
  --submit-interval-ms 0 --no-stop-on-eos --no-prefix-cache \
  --paged-attn --cuda-graph \
  --warmup-runs 1 --repeat-runs 1
```

### Before（只有 TTFT 总数）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 256
Prompt tokens (total): 1024
Completion tokens (total): 16384
Submit wall: 0.026138 s
add_request latency p50/p95/p99: 0.04/0.08/0.13 ms
TTFT p50/p95/p99: 425.47/537.01/537.85 ms
TPOT p50/p95/p99: 19.63/22.57/22.70 ms/token
ITL p50/p95/p99: 18.11/30.13/47.82 ms
Latency p50/p95/p99: 1667.41/1698.27/1698.84 ms
Throughput (completion,total): 9507.13 tokens/s
```

### After（TTFT breakdown）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Paged attention: True
CUDA Graph: True
NVTX: False
Requests: 256
Prompt tokens (total): 1024
Completion tokens (total): 16384
Submit wall: 0.023517 s
add_request latency p50/p95/p99: 0.03/0.06/0.09 ms
Queue wait p50/p95/p99: 404.77/512.30/513.31 ms
Prefill->first token p50/p95/p99: 10.11/11.21/11.24 ms
TTFT p50/p95/p99: 414.69/522.77/523.75 ms
TPOT p50/p95/p99: 19.64/22.51/22.63 ms/token
ITL p50/p95/p99: 18.24/29.59/46.48 ms
Latency p50/p95/p99: 1656.38/1685.28/1685.90 ms
Throughput (completion,total): 9594.75 tokens/s
```

## 结论

这次没有“变快”，但它解决了一个更基础的问题：**把 TTFT 变成可解释的指标**。

从上面的数据可以直接读出：

- TTFT p99 `~523ms` 里，`Queue wait p99 ~513ms`，占了绝大多数
- `Prefill->first token p99 ~11ms`，反而非常小

也就是说：这组配置下 TTFT 的瓶颈主要不是算子本身，而是 **admission/排队**。

下一步如果要把 TTFT 压下去，优先级应该是调度侧：

- 提高/自适应 `prefill_max_batch_size` 或 `prefill_max_tokens`
- 调整 admission policy（fifo vs pack / lookahead）
- 配合 `max_active_requests` 做 backpressure，让系统更稳而不是把 pending 堆爆

有了这两个分量，后面每一次调度策略的改动，都可以用同一套 benchmark 直接回答：**TTFT 变了，是 queue wait 变了，还是 prefill 变了。**

