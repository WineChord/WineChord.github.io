---
classes: wide2
title: "从零实现 LLM Inference：050. Stream Interval（每 N token flush 一次）"
excerpt: "streaming 如果每个 token 都 q.put + HTTP flush，会把 Python/IO overhead 放大。引入 stream_interval：第一段内容立刻发，后续每 N 个 token 合并成一个 chunk 再发，吞吐和 tail latency 都能更稳。"
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

上一版我们加了 `max_inflight_requests` 做回压，把“无限排队”变成“有限排队 + 直接拒绝”，TTFT/Queue wait 的 p99 立刻就收敛了。

但 streaming 还有一个非常现实的开销：**每个 token 都推一个 chunk**。

这条路径里至少有两类固定成本：

- Python 侧：`q.put()` / `q.get()` / 小字符串拼接
- HTTP 侧：chunk 颗粒度越细，flush/调度开销越容易被放大（尤其是高并发）

所以很多工业实现都会提供一个 knob：`stream_interval` / `streaming_interval` —— **不是每个 token 都发，而是每 N 个 token 合并成一个 chunk 再发**。

这一版我们就把这个能力补齐，并且保证：

- **第一段输出不延迟**（TTFT 不被拉长）
- 后续每 N token 合并输出，减少 per-token overhead

## 代码变更

### `roseinfer/server.py`

`SchedulerManager` 新增参数：

- `stream_interval: int = 1`
  - `1`：旧行为（每 token flush）
  - `N>1`：第一段内容立刻 flush，后续每 N 个 token flush 一次（仍然保持顺序）

实现方式是给每个 request 维护一个轻量的 `_StreamState`：

- `buf`: 累积 text piece
- `tokens_since_flush`: 距离上次 flush 的 token 数
- `sent_any`: 是否已经发过第一段（第一段永远立刻发）

核心 diff（删掉无关细节）：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
+class _StreamState:
+    __slots__ = ("buf", "tokens_since_flush", "sent_any")
+    def __init__(self) -> None:
+        self.buf = []
+        self.tokens_since_flush = 0
+        self.sent_any = False
@@
 class SchedulerManager:
     def __init__(..., prefill_max_tokens: Optional[int] = None,
+                 stream_interval: int = 1, ...):
+        self._stream_interval = int(stream_interval)
+        if self._stream_interval <= 0:
+            raise ValueError("stream_interval must be positive")
@@
         self._queues: Dict[int, queue.Queue[Optional[str]]] = {}
         self._detoks: Dict[int, BaseDetokenizer] = {}
+        self._stream_states: Dict[int, _StreamState] = {}
@@
     def add_request(...):
         with self._lock:
             ...
             self._queues[request_id] = q
             self._detoks[request_id] = detok
+            self._stream_states[request_id] = _StreamState()
@@
     def stream_text(...):
         finally:
             with self._lock:
                 self._queues.pop(request_id, None)
                 self._detoks.pop(request_id, None)
+                self._stream_states.pop(request_id, None)
@@
     def _worker_loop(...):
         # decode/prefill 输出 token 时：
         state.tokens_since_flush += 1
         if piece:
             state.buf.append(piece)
         if state.buf and (not state.sent_any or state.tokens_since_flush >= self._stream_interval):
             q.put("".join(state.buf))
             state.buf.clear()
             state.tokens_since_flush = 0
             state.sent_any = True
```

另外把这个 knob 暴露到 server CLI：

- `--stream-interval`（默认 1）

### `roseinfer/benchmark_streaming.py`

新增参数：

- `--stream-interval`：透传到 `SchedulerManager(stream_interval=...)`

并在 summary 里打印出来，方便对比：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
+parser.add_argument("--stream-interval", type=int, default=1, ...)
@@
 print(f"Tokenize workers: {int(args.tokenize_workers)}")
+print(f"Stream interval: {int(args.stream_interval)}")
```

### `tests/test_stream_interval.py`

补一个最小语义测试：

- `stream_interval=1` vs `stream_interval=4`：拼起来的完整文本一致
- `stream_interval=4`：yield 的 piece 数量更少（确实在 batch）

## 运行

```bash
pytest -q
```

```text
............................                                             [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
28 passed, 1 warning in 2.10s
```

## Benchmark（HF GPT-2 / streaming）

这个测试刻意让 decode 的 token 数足够大（把 per-token overhead 放大）：

- `num_requests=64`
- `max_new_tokens=128`
- `max_batch_size=64`：64 个请求全部同时 decode（基本没有排队）
- `--no-stop-on-eos`：避免提前结束
- `--no-prefix-cache`：避免 prefix cache 形态干扰

另外建议加一个 offline 兜底，避免偶发网络抖动：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 1) baseline：`stream_interval=1`

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --num-requests 64 --max-new-tokens 128 \
  --max-batch-size 64 --prefill-max-batch-size 64 \
  --tokenize-workers 4 \
  --no-stop-on-eos --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1
```

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 4
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 0.007343 s
add_request latency p50/p95/p99: 0.01/0.02/0.04 ms
Tokenize p50/p95/p99: 0.08/0.23/0.46 ms
Queue wait (post-tok) p50/p95/p99: 11.84/14.99/15.28 ms
Prefill->first token p50/p95/p99: 23.13/23.21/23.22 ms
TTFT p50/p95/p99: 35.08/38.18/38.68 ms
TPOT p50/p95/p99: 25.63/25.63/25.63 ms/token
ITL p50/p95/p99: 28.52/32.70/33.26 ms
Latency p50/p95/p99: 3292.47/3293.84/3295.95 ms
Throughput (completion,total): 2483.93 tokens/s
```

### 2) 开启 batch flush：`--stream-interval 8`

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --num-requests 64 --max-new-tokens 128 \
  --max-batch-size 64 --prefill-max-batch-size 64 \
  --tokenize-workers 4 \
  --no-stop-on-eos --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1 \
  --stream-interval 8
```

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 4
Stream interval: 8
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 64
Prompt tokens (total): 64
Completion tokens (total): 8192
Submit wall: 0.007699 s
add_request latency p50/p95/p99: 0.01/0.01/0.04 ms
Tokenize p50/p95/p99: 0.11/0.20/1.02 ms
Queue wait (post-tok) p50/p95/p99: 9.66/12.55/12.95 ms
Prefill->first token p50/p95/p99: 23.83/23.91/23.92 ms
TTFT p50/p95/p99: 33.56/36.54/37.65 ms
TPOT p50/p95/p99: 24.02/24.02/24.02 ms/token
ITL p50/p95/p99: 27.54/32.31/32.84 ms
Latency p50/p95/p99: 3086.44/3088.15/3088.77 ms
Throughput (completion,total): 2648.93 tokens/s
```

## 结论

这个 PR 的核心不是“算子更快”，而是 **减少 per-token 的 Python/IO 固定成本**：

- `Throughput`: 2483.93 → 2648.93 tokens/s（+6.6%）
- `Latency p99`: 3295.95 → 3088.77 ms（-6.3%）
- `TPOT`: 25.63 → 24.02 ms/token（更稳）

代价也很明确：客户端收到的 chunk 颗粒度变粗（每 8 token 一段），交互“颗粒感”会变化。所以 `stream_interval` 必须是 knob，而不是强制策略；后面可以进一步做成：根据并发/吞吐自动调节，或者按 request 级别配置。

