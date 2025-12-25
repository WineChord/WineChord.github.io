---
classes: wide2
title: "从零实现 LLM Inference：056. Prefix Cache Radix Tree（longest prefix 查找加速）"
excerpt: "055 做了 longest-prefix reuse，但 longest-prefix 查询还是 O(N) 扫描；这版用 token trie 替换掉 scan，把 cache miss 的 longest-prefix 查找从 ms 级降到 us 级，减少 scheduler CPU overhead。"
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

055 已经把 prefix cache 从 “exact hit” 推到了 “longest-prefix reuse”，但当时为了先跑通逻辑，`find_longest_token_prefix()` 还是 **O(N) 扫描**：

- cache 里 entry 越多，miss 的成本越高
- prompt token 越长（尤其跨过一个 KV block），每次比较还会产生大量 tuple slice 分配

这版就做一件事：**把 longest-prefix 查找从 scan 改成 trie**，外部语义不变（还是“严格前缀”，还是 LRU）。

## 代码变更

### `roseinfer/engine.py`

核心思路：

1) `PrefixCache` 里加一个 `_TokenTrie`，只索引 token-key（`tuple[int, ...]`）的 entry
2) `put/evict/clear` 同步维护 trie（避免悬挂引用）
3) `find_longest_token_prefix()` 直接走 trie：从 root 逐 token 下探，沿路记录最后一个 `entry!=None` 的节点

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
 class PrefixCacheEntry:
@@
         self.last_logits = last_logits.detach().to("cpu")

+class _TokenTrieNode:
+    __slots__ = ("children", "entry", "count")
+    def __init__(self) -> None:
+        self.children: dict[int, "_TokenTrieNode"] = {}
+        self.entry: PrefixCacheEntry | None = None
+        self.count: int = 0
+
+class _TokenTrie:
+    def __init__(self) -> None:
+        self.root = _TokenTrieNode()
+    def insert(self, key: tuple[int, ...], entry: PrefixCacheEntry) -> None:
+        ...
+    def remove(self, key: tuple[int, ...], entry: PrefixCacheEntry | None = None) -> None:
+        ...
+    def longest_prefix(self, key: tuple[int, ...]) -> PrefixCacheEntry | None:
+        node = self.root
+        best: PrefixCacheEntry | None = None
+        for tok in key:
+            nxt = node.children.get(tok)
+            if nxt is None:
+                break
+            node = nxt
+            if node.entry is not None:
+                best = node.entry
+        return best

 class PrefixCache:
     def __init__(...):
         self._entries: OrderedDict[...] = OrderedDict()
+        self._token_trie = _TokenTrie()

     def _evict_one(self) -> None:
         _, entry = self._entries.popitem(last=False)
+        if isinstance(entry.key, tuple):
+            self._token_trie.remove(entry.key, entry)
         self._release_entry(entry)

     def clear(self) -> None:
         while self._entries:
             _, entry = self._entries.popitem(last=False)
+            if isinstance(entry.key, tuple):
+                self._token_trie.remove(entry.key, entry)
             self._release_entry(entry)

     def find_longest_token_prefix(self, key: tuple[int, ...]) -> PrefixCacheEntry | None:
         key_len = len(key)
         if key_len <= 0:
             return None
+        best_entry = self._token_trie.longest_prefix(key)
+        if best_entry is None or int(best_entry.prompt_length) >= key_len:
+            return None
         self._entries.move_to_end(best_entry.key)
         return best_entry

     def put(self, key: PrefixCacheKey, session: "InferenceSession", last_logits: torch.Tensor) -> None:
         ...
         self._entries[key] = entry
+        if isinstance(key, tuple):
+            self._token_trie.insert(key, entry)
         self._entries.move_to_end(key)
```

### `scripts/bench_prefix_cache_lookup.py`

补一个很小的 lookup microbench（只测 `find_longest_token_prefix()` 本身，避免被模型 forward 的噪声淹没）。

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
32 passed, 1 warning in 2.68s
```

## Microbench（lookup）

```bash
python scripts/bench_prefix_cache_lookup.py --num-entries 2048 --key-len 512 --iters 200
```

Before（scan）：

```text
=== prefix cache lookup microbench ===
entries: 2048 key_len: 512
miss: 3185.20 us/lookup
hit : 21.70 us/lookup
```

After（trie）：

```text
=== prefix cache lookup microbench ===
entries: 2048 key_len: 512
miss: 20.32 us/lookup
hit : 20.36 us/lookup
```

**miss 直接从 3.18ms 掉到 20us**（~157x）。这类 “cache 没有可复用前缀” 的 workload，本质上就是在给 scheduler 交 CPU 税，这一版把税收回来了。

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 刻意用 `--unique-prompts` 构造 “**prefix cache 几乎全 miss**” 的场景（但我们仍然会在 miss 时做一次 longest-prefix 查找）。

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats 64 --unique-prompts \
  --pretok --tokenize-workers 0 \
  --num-requests 1024 --max-new-tokens 1 \
  --submit-interval-ms 6 --submit-schedule absolute \
  --max-batch-size 1 --prefill-max-batch-size 1 \
  --prefill-admission-policy fifo \
  --paged-attn --no-stop-on-eos \
  --warmup-runs 0 --repeat-runs 1
```

Before：

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
Requests: 1024
Prompt tokens (total): 68809
Completion tokens (total): 1024
Submit wall: 6.139046 s
Submit interval/schedule: 6.000 ms / absolute
Submit lag p50/p95/p99: 0.06/0.12/14.87 ms
add_request latency p50/p95/p99: 0.03/0.05/0.07 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 0.06/0.27/65.57 ms
Prefill->first token p50/p95/p99: 3.98/4.58/4.77 ms
TTFT p50/p95/p99: 4.09/6.20/70.18 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 4.17/6.47/70.41 ms
Throughput (completion,total): 166.69 tokens/s
```

After：

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
Requests: 1024
Prompt tokens (total): 68809
Completion tokens (total): 1024
Submit wall: 6.138145 s
Submit interval/schedule: 6.000 ms / absolute
Submit lag p50/p95/p99: 0.06/0.08/17.14 ms
add_request latency p50/p95/p99: 0.03/0.05/0.07 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 0.05/0.19/61.77 ms
Prefill->first token p50/p95/p99: 3.79/3.98/4.35 ms
TTFT p50/p95/p99: 3.89/5.27/67.38 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 3.94/5.34/67.62 ms
Throughput (completion,total): 166.70 tokens/s
```

这里最直观的是 TTFT / Prefill->first token 的 p50 都掉了一点（大概 4~5%），说明 “miss 时做 longest-prefix 查找” 的 CPU 账单确实被压下去了。

## 结论

- longest-prefix reuse（055）把功能跑通以后，**下一步就是把查找结构补齐**：scan -> trie/radix tree。
- 这版先用最直观的 trie，把 “O(N) 扫描 + tuple slice” 变成 “O(L) 下探 + dict lookup”。
- 继续往前推的话，就可以考虑 **radix tree / block-trie（按 block_size 压缩边）**，以及更贴近 vLLM 的 block-level prefix cache 组织方式。

