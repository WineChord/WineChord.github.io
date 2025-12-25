---
classes: wide2
title: "从零实现 LLM Inference：054. Prefix Cache Token Key（prompt string -> token ids）"
excerpt: "prefix cache 之前用 prompt 字符串当 key；在 pretok 场景里，prompt 文本不同但 token ids 相同会导致 cache miss。改成优先用 prompt_token_ids tuple 作为 key，并加了一个 benchmark knob 复现/量化收益。"
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

这版解决一个很“真实”的 prefix cache 问题：**我们服务侧经常会把 prompt 预先 tokenize（传 `prompt_token_ids`），但 prompt 文本本身可能被改写/拼接（比如附带 request metadata 的 suffix），导致“文本不同、token ids 一样”。**

在这种情况下，如果 prefix cache 仍然用 `prompt: str` 作为 key：

- cache 本身会 miss（同一个 token 序列被当成不同的 key）
- `OnlineScheduler.add_requests()` 的 in-batch dedup 也会失效（同一个 token 序列会被当成不同 prompt）

这一版的目标很明确：**只要有 `prompt_token_ids`，prefix cache 的 key 就以 token ids 为准**（同时保持对纯 prompt string 路径的兼容）。

## 代码变更

### `roseinfer/engine.py`

核心思路：

1) 引入 `PrefixCacheKey = str | tuple[int, ...]`
2) 只要请求带了 `prompt_token_ids`，就用 `tuple(token_ids[-max_pos:])` 当 key（保持和 truncate 行为一致）
3) `PrefixCache` / `OnlineScheduler.add_requests()` 的 dedup/attach/put 全部改成用 `PrefixCacheKey`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
+PrefixCacheKey = str | tuple[int, ...]
@@
 def _maybe_prefill_with_prefix_cache(..., prompt_token_ids=None):
+    cache_key: PrefixCacheKey = prompt
+    if prompt_token_ids is not None:
+        ids = list(prompt_token_ids)
+        if len(ids) > max_pos:
+            ids = ids[-max_pos:]
+        cache_key = tuple(ids)
     if use_prefix_cache:
-        cached_logits = self.prefix_cache.attach(prompt, session)
+        cached_logits = self.prefix_cache.attach(cache_key, session)
     ...
     if use_prefix_cache:
-        self.prefix_cache.put(prompt, session, last_logits)
+        self.prefix_cache.put(cache_key, session, last_logits)
@@
 class PrefixCache:
-    self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
+    self._entries: OrderedDict[PrefixCacheKey, PrefixCacheEntry] = OrderedDict()
@@
-def attach(self, prompt: str, session): ...
+def attach(self, key: PrefixCacheKey, session): ...
@@
 class OnlineScheduler:
     def add_requests(...):
+        cache_keys: list[PrefixCacheKey] = []
-        first_idx_for_prompt: dict[str, int] = {}
+        first_idx_for_key: dict[PrefixCacheKey, int] = {}
@@
+        cache_key: PrefixCacheKey = req.prompt if req.prompt_token_ids is None else tuple(ids)
+        cache_keys.append(cache_key)
@@
-        src = first_idx_for_prompt.get(req.prompt)
+        src = first_idx_for_key.get(cache_key)
@@
-        cached_logits = eng.prefix_cache.attach(req.prompt, sess)
+        cached_logits = eng.prefix_cache.attach(cache_key, sess)
@@
-        eng.prefix_cache.put(requests[idx].prompt, sessions[idx], logits)
+        eng.prefix_cache.put(cache_keys[idx], sessions[idx], logits)
```

### `roseinfer/benchmark_streaming.py`

为了能稳定复现“prompt 文本不同但 token ids 相同”，加了一个小开关：

- `--pretok-base-token-ids`
  - 只在 `--pretok` 开启时生效
  - tokenization 使用 base prompt（还没 append `--unique-prompts` 的 suffix）
  - 最终效果：**prompt string 每个请求都不同，但 `prompt_token_ids` 完全一致**

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
+  --pretok-base-token-ids
+      tokenize base prompt before applying --unique-prompts
@@
     prompts.append(p)
+    tokenize_prompts.append(base_prompt if args.pretok_base_token_ids else p)
@@
-for p in prompts:
+for p in tokenize_prompts:
     ids = tokenizer.encode(p, add_special_tokens=False)
```

### `tests/test_online_scheduler_add_requests_prefix_cache.py`

补一个最小测试：**prompt 不同，但 `prompt_token_ids` 相同**，应该只 prefill 一次。

```diff
diff --git a/tests/test_online_scheduler_add_requests_prefix_cache.py b/tests/test_online_scheduler_add_requests_prefix_cache.py
@@
+def test_online_scheduler_add_requests_prefix_cache_dedups_token_ids_in_batch() -> None:
+    scheduler.add_requests([
+      OnlineRequest(prompt="a", prompt_token_ids=[1,2,3], ...),
+      OnlineRequest(prompt="b", prompt_token_ids=[1,2,3], ...),
+      OnlineRequest(prompt="c", prompt_token_ids=[1,2,3], ...),
+    ])
+    assert forward_calls == 1
```

## 运行

```bash
pytest -q
```

```text
...............................                                          [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
31 passed, 1 warning in 2.14s
```

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 的关键点是：

- `--unique-prompts`：保证每个请求的 prompt string 都不同（旧实现必然 cache miss）
- `--pretok --pretok-base-token-ids`：保证每个请求的 token ids 都相同（新实现可以 dedup/cache hit）

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --num-requests 32 \
  --max-batch-size 32 --prefill-max-batch-size 32 \
  --max-new-tokens 1 \
  --submit-interval-ms 0 \
  --pretok --pretok-base-token-ids --unique-prompts \
  --no-stop-on-eos
```

### Before（string key）

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Pretok base token ids: True
Tokenize workers: 0
Stream interval: 1
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 32
Prompt tokens (total): 32
Completion tokens (total): 32
Submit wall: 0.084241 s
add_request latency p50/p95/p99: 0.01/0.04/55.23 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 165.74/167.50/167.88 ms
Prefill->first token p50/p95/p99: 20.92/22.20/123.12 ms
TTFT p50/p95/p99: 186.83/188.61/230.09 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 188.49/190.08/230.96 ms
Throughput (completion,total): 117.50 tokens/s
```

### After（token ids key）

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Pretok base token ids: True
Tokenize workers: 0
Stream interval: 1
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 32
Prompt tokens (total): 32
Completion tokens (total): 32
Submit wall: 0.082998 s
add_request latency p50/p95/p99: 0.01/0.03/54.78 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 158.35/159.97/160.21 ms
Prefill->first token p50/p95/p99: 1.90/2.44/111.68 ms
TTFT p50/p95/p99: 160.29/161.70/215.97 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 161.10/162.69/216.78 ms
Throughput (completion,total): 131.18 tokens/s
```

### 结论

- `Prefill->first token p50`：`20.92ms -> 1.90ms`（**~11x**）
- `TTFT p50`：`186.83ms -> 160.29ms`（**-14%**）
- 吞吐：`117.50 -> 131.18 tokens/s`（**+12%**）

TTFT 里最大的项仍然是 `Queue wait`，所以整体 TTFT 的提升没有 `Prefill->first token` 这么夸张；下一步要继续抠 worker loop 的调度/唤醒粒度（让“prefill 很快”真正转化成“TTFT 也很快”）。

