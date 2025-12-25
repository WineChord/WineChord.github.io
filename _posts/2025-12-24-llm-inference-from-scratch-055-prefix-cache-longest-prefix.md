---
classes: wide2
title: "从零实现 LLM Inference：055. Prefix Cache Longest Prefix Reuse（longest prefix KV 复用）"
excerpt: "prefix cache 之前只能 exact hit；这版做 longest-prefix 复用：命中“缓存 prompt 是新 prompt 的前缀”时直接挂载 KV blocks，然后用 decode(T=1) teacher-forcing 补齐 suffix。顺手把 paged-attn Triton autotune 的 cold-start 放到 SchedulerManager 初始化阶段。"
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

这一版继续把 prefix cache 往 vLLM / sglang 那个方向推：**从“只支持 exact hit”变成“支持 longest prefix reuse”。**

现实里很常见的场景是：请求之间共享一个很长的前缀（system prompt / long context / RAG 检索结果），但每个请求在结尾多/少几个 token。只做 exact hit 的 prefix cache，在这种 workload 下收益会被打折。

我们现在的 paged-attn 只支持 decode(`T=1`)，所以这版的实现取舍很清晰：

- **前缀部分**：直接复用缓存里的 KV blocks（不再 prefill）
- **suffix 部分**：用 decode(`T=1`) 的方式把 token 逐个“喂”进去（teacher-forcing），把 KV 补齐到完整 prompt

## 代码变更

### `roseinfer/engine.py`

核心思路：

1) `PrefixCache` 支持 “**找最长的 token-prefix entry**”（不是 trie，先用 O(N) 扫一遍，后面再上 radix tree）
2) `OnlineScheduler.add_requests()` 在 cache miss 时：
   - 如果启用 paged-attn 且 key 是 `prompt_token_ids tuple`
   - 先尝试 `find_longest_token_prefix()`
   - attach 前缀 entry 的 KV
   - 只对 suffix 用 `decode_step_sessions()` 补齐 KV
3) 为了避免第一次 decode 时 Triton autotune 把 TTFT 拉爆，加一个显式 warmup

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
 class InferenceEngine:
+    def warmup_paged_attention_decode(self) -> None:
+        if not self.use_paged_attention:
+            return
+        if self.device.type != \"cuda\" or not torch.cuda.is_available():
+            return
+        token_id = int(self.eos_token_id or 0)
+        sess = InferenceSession(self)
+        sess.prompt_length = 0
+        sess.generated_ids = [token_id]
+        sess.step_count = 1
+        try:
+            self.decode_step_sessions([sess])
+            torch.cuda.synchronize()
+        finally:
+            sess.release_kv_blocks()
+
 class PrefixCache:
+    def find_longest_token_prefix(self, key: tuple[int, ...]) -> PrefixCacheEntry | None:
+        if not self._entries:
+            return None
+        key_len = len(key)
+        if key_len <= 0:
+            return None
+
+        block_size = int(self.kv_manager.block_size)
+        best_entry: PrefixCacheEntry | None = None
+        best_len = 0
+        for entry in reversed(self._entries.values()):  # MRU first
+            if not isinstance(entry.key, tuple):
+                continue
+            entry_len = int(entry.prompt_length)
+            if entry_len <= best_len or entry_len >= key_len:
+                continue
+
+            # 先按 block 比较，再比较最后一个 partial block（最多 63 token）
+            full_blocks = entry_len // block_size
+            ok = True
+            for b in range(full_blocks):
+                start = b * block_size
+                end = start + block_size
+                if entry.key[start:end] != key[start:end]:
+                    ok = False
+                    break
+            if not ok:
+                continue
+            rem = entry_len - full_blocks * block_size
+            if rem > 0:
+                start = full_blocks * block_size
+                end = start + rem
+                if entry.key[start:end] != key[start:end]:
+                    continue
+
+            best_entry = entry
+            best_len = entry_len
+            if best_len >= key_len - 1:
+                break
+        if best_entry is None:
+            return None
+        self._entries.move_to_end(best_entry.key)
+        return best_entry
+
 class OnlineScheduler:
     def add_requests(...):
+        prefix_suffix_ids: dict[int, list[int]] = {}
@@
         if self.use_prefix_cache:
             cached_logits = eng.prefix_cache.attach(cache_key, sess)
             if cached_logits is not None:
                 last_logits_per_req[i] = cached_logits
                 continue
+            if eng.use_paged_attention and isinstance(cache_key, tuple):
+                prefix = eng.prefix_cache.find_longest_token_prefix(cache_key)
+                if prefix is not None and prefix.prompt_length < len(ids):
+                    eng.prefix_cache.attach(prefix.key, sess)
+                    prefix_suffix_ids[i] = ids[prefix.prompt_length:]
+                    continue
@@
+    if prefix_suffix_ids:
+        # 对 suffix 做 teacher-forcing：decode(T=1) 把 KV 补齐到完整 prompt
+        suffix_pos: dict[int, int] = {idx: 0 for idx in prefix_suffix_ids}
+        while True:
+            active = [
+                idx
+                for idx, pos in suffix_pos.items()
+                if pos < len(prefix_suffix_ids[idx])
+            ]
+            if not active:
+                break
+            active_sessions = [sessions[idx] for idx in active]
+            for idx, sess in zip(active, active_sessions):
+                tok = int(prefix_suffix_ids[idx][suffix_pos[idx]])
+                sess.generated_ids = [tok]
+                sess.step_count = 1
+            step_logits = eng.decode_step_sessions(active_sessions)
+            for b, idx in enumerate(active):
+                logits = step_logits[b : b + 1]
+                last_logits_per_req[idx] = logits
+                sessions[idx].prompt_length += 1
+                suffix_pos[idx] += 1
+
+        # 补齐后把完整 prompt 放回 prefix cache（后续就能 exact hit 了）
+        for idx in prefix_suffix_ids:
+            sess = sessions[idx]
+            sess.generated_ids = []
+            sess.step_count = 0
+            logits = last_logits_per_req[idx]
+            if logits is None:
+                raise RuntimeError(f"missing prefix reuse logits for request {rids[idx]}")
+            if self.use_prefix_cache:
+                eng.prefix_cache.put(cache_keys[idx], sess, logits)
```

### `roseinfer/server.py`

把 warmup 放到 `SchedulerManager` 初始化阶段（只在 `paged-attn + cuda` 时生效），把 autotune 从“首个真实请求”挪到“服务启动阶段”。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
 class SchedulerManager:
     def __init__(...):
         self.scheduler = OnlineScheduler(...)
+        self.engine.warmup_paged_attention_decode()
```

### `tests/test_prefix_cache_longest_prefix.py`

补一个最小 GPU 测试：base prompt (64 tokens) 先进入 cache，extended prompt (65 tokens) 应该走 **decode(T=1)** 的补齐路径（也就是 `model.forward` 只看到 `seq_len==1`）。

## 运行

```bash
pytest -q
```

```text
................................                                         [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
32 passed, 1 warning in 2.65s
```

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 刻意构造“**prompt 之间是严格前缀关系**”：

- 16 个请求的 prompt 长度分别是 `900..915`
- `prefill-max-batch-size=1`（一个个进）
- 所以理想行为是：
  - 第一个请求做一次完整 prefill
  - 后续请求全部走 prefix reuse：只需要补 1 个 token 的 suffix

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" \
  --prompt-repeats "900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915" \
  --num-requests 16 \
  --max-batch-size 1 --prefill-max-batch-size 1 \
  --prefill-admission-policy fifo \
  --max-new-tokens 1 \
  --submit-interval-ms 0 \
  --pretok --tokenize-workers 0 \
  --paged-attn \
  --no-stop-on-eos
```

### Before（无 longest-prefix reuse）

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Pretok base token ids: False
Tokenize workers: 0
Stream interval: 1
Paged attention: True
CUDA Graph: False
NVTX: False
Requests: 16
Prompt tokens (total): 14520
Completion tokens (total): 16
Submit wall: 0.081218 s
add_request latency p50/p95/p99: 0.02/19.59/66.47 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 275.23/382.11/391.26 ms
Prefill->first token p50/p95/p99: 16.49/54.81/146.22 ms
TTFT p50/p95/p99: 291.81/397.50/406.67 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 292.23/397.85/406.86 ms
Throughput (completion,total): 32.63 tokens/s
```

### After（longest-prefix reuse + paged-attn warmup）

```text
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Pretok base token ids: False
Tokenize workers: 0
Stream interval: 1
Paged attention: True
CUDA Graph: False
NVTX: False
Requests: 16
Prompt tokens (total): 14520
Completion tokens (total): 16
Submit wall: 0.081579 s
add_request latency p50/p95/p99: 0.02/19.76/67.06 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 76.31/100.59/102.73 ms
Prefill->first token p50/p95/p99: 3.64/16.53/46.09 ms
TTFT p50/p95/p99: 83.49/113.28/128.69 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 83.65/113.43/129.05 ms
Throughput (completion,total): 84.91 tokens/s
```

### 结论

- `Prefill->first token p50`：`16.49ms -> 3.64ms`（**~4.5x**）
- `TTFT p50`：`291.81ms -> 83.49ms`（**-71%**）
- 吞吐：`32.63 -> 84.91 tokens/s`（**~2.6x**）

这组数据里，`Queue wait` 的下降最关键：原来每个请求都要做完整 prefill，worker loop 一次只能吃一个请求（`prefill-max-batch-size=1`），后面的请求只能排队；现在除了第一个请求外，后续基本都在 “prefix reuse + 1 token 补齐” 里结束 prefill 阶段，所以排队时间大幅缩短。

下一步如果要继续往业界靠：

- prefix cache 的查找从 O(N) 扫描升级成 radix tree / block trie（把前缀查找从“扫表”变成“走树”）
- paged-attn 做 chunked prefill（解决 suffix 很长时 `decode(T=1)` 需要循环的问题）
