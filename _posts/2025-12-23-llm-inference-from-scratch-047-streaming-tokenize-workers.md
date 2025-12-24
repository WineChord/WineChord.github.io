---
classes: wide2
title: "从零实现 LLM Inference：047. Streaming Tokenize Workers（把 encode 挪出 add_request）"
excerpt: "add_request 里同步做 tokenizer.encode 会放大 submit wall / p99。引入 tokenize_workers 线程池，把 tokenization 变成后台阶段，并把 TTFT 拆到 tokenize/queue/prefill 三段。"
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

上一版我们加了 `--pretok`，能把 tokenization 从提交路径里挪出去，`Submit wall`/`add_request latency p99` 都会立刻变好。

但 `--pretok` 更像是 benchmark/客户端技巧：真实服务形态里，入口通常拿到的是 **prompt string**，tokenization 依然需要做。

这一版更偏“服务端视角”：**把 `tokenizer.encode()` 从 `add_request()` 线程里移出去**，让提交路径变成轻量的 enqueue；tokenization 在后台 worker 里做，完成后再进入 pending queue。

同时，为了让指标更可解释，把 TTFT breakdown 再往前拆一段：多一个 `Tokenize` 阶段。

## 代码变更

### `roseinfer/server.py`

`SchedulerManager` 新增一个参数：

- `tokenize_workers: int = 0`
  - `0`：保持旧行为（`add_request()` 内同步 `tokenizer.encode()`）
  - `>0`：`add_request()` 只做 request id/queue/detok 的初始化 + enqueue `_TokenizeTask`，tokenization 在后台线程做完后再入 `_pending`

并新增：

- `_tokenize_timestamps`：记录每个请求“tokenization 完成”的时间戳（仅在 `record_token_timestamps=True` 时写入）
- `pop_tokenize_timestamp()`：给 benchmark 拉数据

核心 diff（删掉无关细节）：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
 class SchedulerManager:
     def __init__(
         self,
         engine: InferenceEngine,
@@
         record_token_timestamps: bool = False,
+        tokenize_workers: int = 0,
@@
+        self._tokenize_workers = int(tokenize_workers)
+        self._tokenize_q = queue.Queue() if self._tokenize_workers > 0 else None
+        self._tokenize_threads = [...]
+        self._tokenize_timestamps: Dict[int, float] = {}
+
+        if self._tokenize_q is not None:
+            for ...:
+                threading.Thread(target=self._tokenize_loop, ...).start()
@@
     def add_request(..., prompt_token_ids: Optional[list[int]] = None, ...) -> int:
         with self._lock:
             detok = self.engine._make_detok()
             request_id = ...
             self._queues[request_id] = queue.Queue()
             self._detoks[request_id] = detok
@@
+        # async tokenize path
+        if prompt_token_ids is None and self._tokenize_q is not None:
+            self._tokenize_q.put(_TokenizeTask(request_id=..., prompt=..., ...))
+            self._wakeup.set()
+            return request_id
+
         # sync tokenize path (old behavior)
         token_ids = tokenizer.encode(prompt, add_special_tokens=False)
         ...
         detok.start_prompt(token_ids)
+        if self._record_token_timestamps:
+            self._tokenize_timestamps[request_id] = time.perf_counter()
         self._pending.put(_PendingRequest(...))
@@
+    def pop_tokenize_timestamp(self, request_id: int) -> float | None:
+        ...
+
+    def _tokenize_loop(self) -> None:
+        task = self._tokenize_q.get()
+        token_ids = tokenizer.encode(task.prompt, add_special_tokens=False)
+        ...
+        detok.start_prompt(token_ids)
+        if record_token_timestamps: self._tokenize_timestamps[rid] = time.perf_counter()
+        self._pending.put(_PendingRequest(...))
+        self._wakeup.set()
```

### `roseinfer/benchmark_streaming.py`

新增 `--tokenize-workers`，并把 TTFT breakdown 拆成三段：

- `Tokenize`：`submit_end -> tokenize_ts`
- `Queue wait (post-tok)`：`tokenize_ts -> admit_ts`
- `Prefill->first token`：`admit_ts -> first_token_ts`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
     parser.add_argument("--pretok", action="store_true", ...)
+    parser.add_argument("--tokenize-workers", type=int, default=0, ...)
@@
 class StreamResult:
     submit_end: float
+    tokenize_ts: float
     admit_ts: float
@@
     tokenize_ts = mgr.pop_tokenize_timestamp(request_id) or submit_end
     admit_ts = mgr.pop_admit_timestamp(request_id) or tokenize_ts
@@
     tokenize_lats = [r.tokenize_ts - r.submit_end ...]
     queue_waits = [r.admit_ts - r.tokenize_ts ...]
@@
     print(f"Tokenize workers: {args.tokenize_workers}")
+    print(f"Tokenize p50/p95/p99: ...")
+    print(f"Queue wait (post-tok) p50/p95/p99: ...")
```

### `tests/test_server_pretok.py`

加一个行为测试，保证 `tokenize_workers>0` 时 `add_request()` 不会被 `encode()` 阻塞（用 gate 把 encode 卡住）：

```diff
diff --git a/tests/test_server_pretok.py b/tests/test_server_pretok.py
@@
+def test_server_tokenize_workers_does_not_block_add_request() -> None:
+    gate = threading.Event()
+    tok = _BlockingTokenizer(gate=gate)
+    mgr = SchedulerManager(engine, max_batch_size=1, tokenize_workers=1)
+    rid = mgr.add_request("hello", max_new_tokens=1, stop_on_eos=False)
+    assert tok.encode_calls == 0
+    gate.set()
+    _ = list(mgr.stream_text(rid))
+    assert tok.encode_calls == 1
```

## 运行

```bash
pytest -q
```

```text
..........................                                               [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/rosellm/.conda/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
26 passed, 1 warning in 2.09s
```

## Benchmark（HF GPT-2 / streaming）

这版主要看两个点：

- `Submit wall`：一次性灌满请求时，提交路径的吞吐
- `add_request latency p99`：服务端入口的 per-request overhead

命令（baseline：`tokenize_workers=0`）：

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats 256 --unique-prompts \
  --num-requests 256 --max-new-tokens 1 \
  --max-batch-size 16 --prefill-max-batch-size 16 \
  --submit-interval-ms 0 --no-stop-on-eos --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1 \
  --tokenize-workers 0
```

### Before（`tokenize_workers=0`）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Tokenize workers: 0
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.079206 s
add_request latency p50/p95/p99: 0.24/0.28/0.35 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 464.83/983.04/986.32 ms
Prefill->first token p50/p95/p99: 69.47/71.33/71.71 ms
TTFT p50/p95/p99: 534.49/1048.12/1051.12 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 534.88/1048.88/1051.76 ms
Throughput (completion,total): 227.04 tokens/s
```

命令（after：`tokenize_workers=4`）：

```bash
... --tokenize-workers 4
```

### After（`tokenize_workers=4`）

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
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.029888 s
add_request latency p50/p95/p99: 0.01/0.02/0.05 ms
Tokenize p50/p95/p99: 0.32/0.57/0.83 ms
Queue wait (post-tok) p50/p95/p99: 505.24/1048.71/1049.88 ms
Prefill->first token p50/p95/p99: 69.35/70.24/70.52 ms
TTFT p50/p95/p99: 574.49/1107.87/1108.84 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 575.42/1107.97/1109.15 ms
Throughput (completion,total): 224.89 tokens/s
```

## 结论

这版的核心收益是“入口变轻”，可以用数据直接看到：

- `Submit wall`：`79.2ms -> 29.9ms`（约 `2.6x`）
- `add_request latency p99`：`0.35ms -> 0.05ms`（约 `7x`）

同时我们把 tokenization 从黑盒里拆出来了：

- `Tokenize p99 ~0.83ms`（现在能在同一份输出里读到它）
- 这组配置下 TTFT 依然主要被 `Queue wait (post-tok) + Prefill->first token` 主导（tokenize 只是很小的一段）

最后，TTFT 在这个 “burst submit” 配置下会跟着变大，是因为提交更快会把请求更早地灌进系统，排队更深；如果要对比 TTFT，最好固定 arrival pattern（比如用 `--submit-interval-ms` 控制到达节奏），不然 TTFT 的变化很容易混进“负载形态变化”的影响。
