---
classes: wide2
title: "从零实现 LLM Inference：029. Prefix Cache + Prefill Micro-Batching"
excerpt: "prefix cache 开启时也能 micro-batch prefill：hit 直接 attach，miss 合并成一次 forward。"
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

上一版（Prefill Micro-Batching）把 streaming admission 的 prefill 合并成一次 batched forward，TTFT/吞吐都有明显收益。

但那一版为了让 PR 足够小：**prefix cache 开启时，`add_requests()` 会 fallback 到逐条 `add_request()`**。prefix cache hit 没问题，miss 就会退化成串行 prefill（burst 场景 TTFT/尾延迟被拉爆）。

这次 mini PR 把这条链路补齐：**prefix cache 开启时也能 batch prefill（hit/miss 分流 + 同 prompt 去重）**，并顺手把 streaming benchmark 的指标补齐到 TTFT/TPOT/ITL（p99）。

## 代码变更

这次一共动了四块：

1. `OnlineScheduler.add_requests()`：prefix cache 开启时不再 fallback，miss 走一次 batched prefill。
2. batch admission 里做 **同 prompt 去重**：同一轮只 prefill 一次，其它 request 直接共享 blocks + logits。
3. `benchmark_streaming` 补齐 **TPOT/ITL**，并把所有关键指标统一打印 **p50/p95/p99**。
4. 加一个最小单测：确保 prefix cache 开启时 `add_requests()` 仍然只 forward 一次。

### `roseinfer/engine.py`

核心逻辑拆成三段（顺序很关键）：

1. **Prefix cache hit**：直接 `attach(prompt, session)`，拿到 `last_logits` + 复用 KV blocks（不跑 prefill）。
2. **Prefix cache miss**：把 miss 的请求合成一次 batched prefill forward，再逐条把 KV 写入 block manager；同时 `prefix_cache.put()` 存进去。
3. **同 prompt 去重**：同一批 admission 里如果出现相同 prompt，只 prefill 一次，其它 request 共享 blocks + logits（避免“同 prompt burst 反而变慢”）。

另外：为了避免采样（`torch.multinomial`）顺序变化导致的随机性差异，**第一个 token 的 sampling 统一按请求原始顺序逐个做**。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 78ba210..b981dee 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1574,21 +1574,6 @@ class OnlineScheduler:
     ) -> list[int]:
         if not requests:
             return []
-        if self.use_prefix_cache:
-            return [
-                self.add_request(
-                    prompt=req.prompt,
-                    max_new_tokens=req.max_new_tokens,
-                    temperature=req.temperature,
-                    top_k=req.top_k,
-                    top_p=req.top_p,
-                    stop_on_eos=req.stop_on_eos,
-                    do_sample=req.do_sample,
-                    prompt_token_ids=req.prompt_token_ids,
-                    request_id=req.request_id,
-                )
-                for req in requests
-            ]
 
         eng = self.engine
         eng.model.eval()
@@ -1616,7 +1601,12 @@ class OnlineScheduler:
 
         sessions: list[InferenceSession] = []
         token_ids_list: list[list[int]] = []
-        for req in requests:
+        last_logits_per_req: list[torch.Tensor | None] = [None for _ in requests]
+        miss_idx: list[int] = []
+        dup_of: dict[int, int] = {}
+        first_idx_for_prompt: dict[str, int] = {}
+
+        for i, req in enumerate(requests):
             rid = alloc_rid(req.request_id)
             rids.append(rid)
 
@@ -1656,37 +1646,84 @@ class OnlineScheduler:
             )
             sessions.append(sess)
 
-        batch_idx = [i for i, s in enumerate(sessions) if not s.finished]
-        if batch_idx:
-            batch_token_ids = [token_ids_list[i] for i in batch_idx]
+            if self.use_prefix_cache:
+                src = first_idx_for_prompt.get(req.prompt)
+                if src is not None:
+                    dup_of[i] = src
+                    continue
+                first_idx_for_prompt[req.prompt] = i
+
+                cached_logits = eng.prefix_cache.attach(req.prompt, sess)
+                if cached_logits is not None:
+                    last_logits_per_req[i] = cached_logits
+                    continue
+
+            miss_idx.append(i)
+
+        if miss_idx:
+            batch_token_ids = [token_ids_list[i] for i in miss_idx]
             input_ids, attn_mask, lengths, _ = eng._encode_prompt_token_ids_batch(
                 batch_token_ids
             )
-            batch_sessions = [sessions[i] for i in batch_idx]
+            batch_sessions = [sessions[i] for i in miss_idx]
             last_logits = eng._prefill_register_kv_batch(
                 sessions=batch_sessions,
                 input_ids=input_ids,
                 attention_mask=attn_mask,
                 lengths=lengths,
             )
-            for b, sess in enumerate(batch_sessions):
-                token_id = eng._sample_next_token(
-                    logits=last_logits[b : b + 1],
-                    temperature=sess.temperature,
-                    top_k=sess.top_k,
-                    top_p=sess.top_p,
-                    do_sample=sess.do_sample,
-                )
-                sess.generated_ids.append(int(token_id))
-                sess.step_count = 1
-                if sess.stop_on_eos:
-                    eos_id = eng.eos_token_id
-                    if eos_id is not None and int(token_id) == eos_id:
-                        sess.finished = True
-                if sess.max_new_tokens > 0 and sess.step_count >= sess.max_new_tokens:
-                    sess.finished = True
+            for b, idx in enumerate(miss_idx):
+                logits = last_logits[b : b + 1]
+                last_logits_per_req[idx] = logits
+                if self.use_prefix_cache:
+                    eng.prefix_cache.put(
+                        requests[idx].prompt,
+                        sessions[idx],
+                        logits,
+                    )
+
+        if dup_of:
+            kvm = eng.kv_manager
+            for idx, src in dup_of.items():
+                sess = sessions[idx]
+                if sess.finished:
+                    continue
+                src_sess = sessions[src]
+                if src_sess.finished:
+                    sess.finished = True
+                    continue
+                sess.prompt_length = src_sess.prompt_length
+                sess.block_ids_per_layer = [[] for _ in range(kvm.num_layers)]
+                for layer_idx, block_ids in enumerate(src_sess.block_ids_per_layer):
+                    if not block_ids:
+                        continue
+                    kvm.incref_blocks(block_ids)
+                    sess.block_ids_per_layer[layer_idx] = list(block_ids)
+                last_logits_per_req[idx] = last_logits_per_req[src]
+
+        for idx, sess in enumerate(sessions):
+            if sess.finished:
+                continue
+            logits = last_logits_per_req[idx]
+            if logits is None:
+                raise RuntimeError(f"missing prefill logits for request {rids[idx]}")
+            token_id = eng._sample_next_token(
+                logits=logits,
+                temperature=sess.temperature,
+                top_k=sess.top_k,
+                top_p=sess.top_p,
+                do_sample=sess.do_sample,
+            )
+            sess.generated_ids.append(int(token_id))
+            sess.step_count = 1
+            if sess.stop_on_eos:
+                eos_id = eng.eos_token_id
+                if eos_id is not None and int(token_id) == eos_id:
+                    sess.finished = True
+            if sess.max_new_tokens > 0 and sess.step_count >= sess.max_new_tokens:
+                sess.finished = True
+            if sess.finished:
+                sess.release_kv_blocks()
 
         for rid, sess in zip(rids, sessions):
             self._sessions[rid] = sess
```

### `roseinfer/server.py`

为了算 TPOT/ITL，需要“每个 token 真正被推到 queue 的时间戳”。最简单的做法就是在 worker 里记录 `time.perf_counter()`。

默认不打开（避免影响正常 server）；`benchmark_streaming` 显式开启。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 8479835..b84fafa 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -111,6 +111,7 @@ class SchedulerManager:
         self,
         engine: InferenceEngine,
         max_batch_size: int = 8,
+        record_token_timestamps: bool = False,
     ) -> None:
@@ -121,6 +122,8 @@ class SchedulerManager:
         self._wakeup = threading.Event()
         self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
         self._detoks: Dict[int, BaseDetokenizer] = {}
+        self._record_token_timestamps = bool(record_token_timestamps)
+        self._token_timestamps: Dict[int, list[float]] = {}
@@ -177,6 +181,8 @@ class SchedulerManager:
             q: "queue.Queue[Optional[str]]" = queue.Queue()
             self._queues[request_id] = q
             self._detoks[request_id] = detok
+            if self._record_token_timestamps:
+                self._token_timestamps[request_id] = []
@@ -193,6 +199,14 @@ class SchedulerManager:
         self._wakeup.set()
         return request_id
+
+    def pop_token_timestamps(
+        self,
+        request_id: int,
+    ) -> list[float]:
+        with self._lock:
+            out = self._token_timestamps.pop(request_id, None)
+        return list(out) if out is not None else []
@@ -249,10 +263,17 @@ class SchedulerManager:
                     with self._lock:
                         q = self._queues.get(rid)
                         detok = self._detoks.get(rid)
+                        token_ts = (
+                            self._token_timestamps.get(rid)
+                            if self._record_token_timestamps
+                            else None
+                        )
@@ -257,6 +278,8 @@ class SchedulerManager:
                     if q is None or detok is None:
                         self.scheduler.discard_request(rid)
                         continue
                     for tid in self.scheduler.get_generated_ids(rid):
+                        if token_ts is not None:
+                            token_ts.append(time.perf_counter())
                         piece = detok.on_token(int(tid))
                         if piece:
                             q.put(piece)
@@ -274,9 +295,16 @@ class SchedulerManager:
                     with self._lock:
                         q = self._queues.get(rid)
                         detok = self._detoks.get(rid)
+                        token_ts = (
+                            self._token_timestamps.get(rid)
+                            if self._record_token_timestamps
+                            else None
+                        )
@@ -281,6 +309,8 @@ class SchedulerManager:
                     if q is None or detok is None:
                         self.scheduler.discard_request(rid)
                         continue
+                    if token_ts is not None:
+                        token_ts.append(time.perf_counter())
                     piece = detok.on_token(int(token_id))
                     if piece:
                         q.put(piece)
```

### `roseinfer/benchmark_streaming.py`

TPOT/ITL 直接从 `token_timestamps` 算，所有指标都打印 p50/p95/p99。

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
index e44daed..6462272 100644
--- a/rosellm/roseinfer/benchmark_streaming.py
+++ b/rosellm/roseinfer/benchmark_streaming.py
@@ -23,6 +23,7 @@ class StreamResult:
     finish_ts: float
     completion_text: str
     completion_tokens: int
+    token_timestamps: list[float]
@@ -178,7 +179,11 @@ def main() -> None:
         kv_cache_max_concurrency=kv_cache_max_concurrency,
         prefix_cache_max_entries=len(set(prompts)),
     )
-    mgr = SchedulerManager(engine, max_batch_size=int(args.max_batch_size))
+    mgr = SchedulerManager(
+        engine,
+        max_batch_size=int(args.max_batch_size),
+        record_token_timestamps=True,
+    )
@@ -194,14 +199,15 @@ def main() -> None:
         first_token_ts: float | None = None
         pieces: list[str] = []
         for piece in mgr.stream_text(request_id):
-            if first_token_ts is None:
-                first_token_ts = time.perf_counter()
             pieces.append(piece)
         finish_ts = time.perf_counter()
+        token_ts = mgr.pop_token_timestamps(request_id)
+        if token_ts:
+            first_token_ts = token_ts[0]
         if first_token_ts is None:
             first_token_ts = finish_ts
         completion_text = "".join(pieces)
-        completion_tokens = count_tokens(engine.tokenizer, completion_text)
+        completion_tokens = len(token_ts)
@@ -244,8 +251,18 @@ def main() -> None:
         add_lats = [r.submit_end - r.submit_start for r in results]
         ttfts = [r.first_token_ts - r.submit_start for r in results]
         totals = [r.finish_ts - r.submit_start for r in results]
-        completion_tokens = [r.completion_tokens for r in results]
-        sum_tokens = sum(completion_tokens)
+        completion_tokens = [int(r.completion_tokens) for r in results]
+        sum_tokens = int(sum(completion_tokens))
+
+        tpots: list[float] = []
+        itls: list[float] = []
+        for r in results:
+            ts = r.token_timestamps
+            if len(ts) < 2:
+                continue
+            tpots.append((ts[-1] - ts[0]) / float(len(ts) - 1))
+            for i in range(1, len(ts)):
+                itls.append(ts[i] - ts[i - 1])
@@ -270,6 +287,20 @@ def main() -> None:
             f"{percentile(ttfts, 95)*1e3:.2f}/"
             f"{percentile(ttfts, 99)*1e3:.2f} ms"
         )
+        tpot_p50 = statistics.median(tpots) if tpots else 0.0
+        itl_p50 = statistics.median(itls) if itls else 0.0
+        print(
+            f"TPOT p50/p95/p99: "
+            f"{tpot_p50*1e3:.2f}/"
+            f"{percentile(tpots, 95)*1e3:.2f}/"
+            f"{percentile(tpots, 99)*1e3:.2f} ms/token"
+        )
+        print(
+            f"ITL p50/p95/p99: "
+            f"{itl_p50*1e3:.2f}/"
+            f"{percentile(itls, 95)*1e3:.2f}/"
+            f"{percentile(itls, 99)*1e3:.2f} ms"
+        )
```

### 新增测试

主要覆盖两件事：

- prefix cache 开启时 `add_requests()` 不再 fallback 逐条 prefill（forward 只跑一次）
- 同 prompt 在同一批 admission 里会去重（forward 仍然只跑一次）

文件：`tests/test_online_scheduler_add_requests_prefix_cache.py`

```diff
diff --git a/tests/test_online_scheduler_add_requests_prefix_cache.py b/tests/test_online_scheduler_add_requests_prefix_cache.py
new file mode 100644
index 0000000..830ea02
--- /dev/null
+++ b/tests/test_online_scheduler_add_requests_prefix_cache.py
@@ -0,0 +1,156 @@
+import torch
+
+from rosellm.roseinfer.engine import InferenceEngine, OnlineRequest, OnlineScheduler
+from rosellm.rosetrainer.config import GPTConfig
+from rosellm.rosetrainer.model import GPTModel
+
+
+class _CountingTokenizer:
+    def __init__(self, vocab_size: int = 128) -> None:
+        self.vocab_size = int(vocab_size)
+        self.eos_token_id = 0
+        self.pad_token_id = 0
+        self.encode_calls = 0
+
+    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
+        self.encode_calls += 1
+        del text, add_special_tokens
+        return [1, 2, 3]
+
+    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
+        del ids, skip_special_tokens
+        return ""
+
+
+def test_online_scheduler_add_requests_prefix_cache_batches_prefill() -> None:
+    torch.manual_seed(0)
+    cfg = GPTConfig(
+        vocab_size=128,
+        max_position_embeddings=32,
+        n_layers=2,
+        n_heads=2,
+        d_model=32,
+        d_ff=64,
+        dropout=0.0,
+    )
+    tok = _CountingTokenizer(vocab_size=128)
+    model = GPTModel(cfg)
+    forward_calls = 0
+    orig_forward = model.forward
+
+    def counting_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
+        nonlocal forward_calls
+        forward_calls += 1
+        return orig_forward(*args, **kwargs)
+
+    model.forward = counting_forward  # type: ignore[method-assign]
+
+    engine = InferenceEngine(
+        model=model,
+        config=cfg,
+        tokenizer=tok,
+        tokenizer_name="dummy",
+        device="cpu",
+        use_amp=False,
+        kv_cache_max_concurrency=8,
+        prefix_cache_max_entries=8,
+    )
+
+    scheduler = OnlineScheduler(engine, max_batch_size=8, use_prefix_cache=True)
+    scheduler.add_requests(
+        [
+            OnlineRequest(
+                prompt="p0",
+                prompt_token_ids=[1, 2, 3],
+                max_new_tokens=1,
+                stop_on_eos=False,
+                do_sample=False,
+                request_id=0,
+            ),
+            OnlineRequest(
+                prompt="p1",
+                prompt_token_ids=[1, 2, 3, 4],
+                max_new_tokens=1,
+                stop_on_eos=False,
+                do_sample=False,
+                request_id=1,
+            ),
+            OnlineRequest(
+                prompt="p2",
+                prompt_token_ids=[1, 2],
+                max_new_tokens=1,
+                stop_on_eos=False,
+                do_sample=False,
+                request_id=2,
+            ),
+        ]
+    )
+    assert tok.encode_calls == 0
+    assert forward_calls == 1
+
+
+def test_online_scheduler_add_requests_prefix_cache_dedups_prompts_in_batch() -> None:
+    torch.manual_seed(0)
+    cfg = GPTConfig(
+        vocab_size=128,
+        max_position_embeddings=32,
+        n_layers=2,
+        n_heads=2,
+        d_model=32,
+        d_ff=64,
+        dropout=0.0,
+    )
+    tok = _CountingTokenizer(vocab_size=128)
+    model = GPTModel(cfg)
+    forward_calls = 0
+    orig_forward = model.forward
+
+    def counting_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
+        nonlocal forward_calls
+        forward_calls += 1
+        return orig_forward(*args, **kwargs)
+
+    model.forward = counting_forward  # type: ignore[method-assign]
+
+    engine = InferenceEngine(
+        model=model,
+        config=cfg,
+        tokenizer=tok,
+        tokenizer_name="dummy",
+        device="cpu",
+        use_amp=False,
+        kv_cache_max_concurrency=8,
+        prefix_cache_max_entries=8,
+    )
+
+    scheduler = OnlineScheduler(engine, max_batch_size=8, use_prefix_cache=True)
+    scheduler.add_requests(
+        [
+            OnlineRequest(
+                prompt="same",
+                prompt_token_ids=[1, 2, 3],
+                max_new_tokens=1,
+                stop_on_eos=False,
+                do_sample=False,
+                request_id=0,
+            ),
+            OnlineRequest(
+                prompt="same",
+                prompt_token_ids=[1, 2, 3],
+                max_new_tokens=1,
+                stop_on_eos=False,
+                do_sample=False,
+                request_id=1,
+            ),
+            OnlineRequest(
+                prompt="same",
+                prompt_token_ids=[1, 2, 3],
+                max_new_tokens=1,
+                stop_on_eos=False,
+                do_sample=False,
+                request_id=2,
+            ),
+        ]
+    )
+    assert tok.encode_calls == 0
+    assert forward_calls == 1
```

## 指标口径

这版 benchmark 里三个指标的口径：

- TTFT：`t_first_token - t_submit`
- TPOT：`(t_last_token - t_first_token) / (n_tokens - 1)`
- ITL：`t_i - t_{i-1}`（把所有 token 的间隔摊平到一个分布，看 p50/p95/p99）

这里的 `t_first_token / t_i` 是在 worker 里 **token 真正被推到 streaming queue** 的时刻打点，所以它包含了：

- prefill 在 worker 里占用的时间
- worker 的调度/锁/队列开销
- decode step 之间的空洞

所以这次虽然没改 decode kernel，**TPOT/ITL 也可能变好**：之前 miss 串行 prefill 会把一整段时间塞在 worker 里，decode step 被挡住，token gap 自然变大；现在 miss 合并成一次 batched prefill，这段阻塞就收敛了。

## 运行

### 单测

```shell
pytest -q
```

输出：

```text
.........                                                                [100%]
9 passed, 1 warning in 1.63s
```

### Benchmark（HF GPT-2）

同一个命令，直接对比（离线跑可以加上 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`）：

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --unique-prompts \
  --num-requests 32 \
  --max-new-tokens 8 \
  --no-stop-on-eos
```

### Before（prefix cache 开启，但 miss 串行 prefill）

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.084512 s
add_request latency p50/p95/p99: 0.05/0.14/54.01 ms
TTFT p50/p95/p99: 418.48/798.66/798.77 ms
TPOT p50/p95/p99: 140.64/161.01/165.49 ms/token
ITL p50/p95/p99: 107.39/270.98/400.98 ms
Latency p50/p95/p99: 1415.70/1478.94/1479.14 ms
Throughput (completion,total): 163.84 tokens/s
```

### After（prefix cache hit/miss 分流 + miss 合并 batched prefill）

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.082502 s
add_request latency p50/p95/p99: 0.04/0.16/52.72 ms
TTFT p50/p95/p99: 143.41/260.17/260.26 ms
TPOT p50/p95/p99: 100.38/102.51/106.35 ms/token
ITL p50/p95/p99: 101.38/126.63/150.47 ms
Latency p50/p95/p99: 862.18/927.23/927.38 ms
Throughput (completion,total): 253.68 tokens/s
```

对比一下核心指标：

- TTFT p50：`418.48ms -> 143.41ms`（~2.9x）
- TTFT p95：`798.66ms -> 260.17ms`（~3.1x）
- TPOT p50：`140.64ms -> 100.38ms / token`（~1.4x）
- ITL p99：`400.98ms -> 150.47ms`（~2.7x）
- Latency p50：`1415.70ms -> 862.18ms`（~1.6x）
- Throughput：`163.84 -> 253.68 tokens/s`（~1.5x）
