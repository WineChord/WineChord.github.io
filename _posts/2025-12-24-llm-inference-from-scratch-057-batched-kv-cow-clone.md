---
classes: wide2
title: "从零实现 LLM Inference：057. Batched KV COW（shared block 的 append_token_batch fast path）"
excerpt: "prefix cache 复用会让 decode 的最后一个 KV block refcount>1；之前 append_token_batch 直接退化成逐 request 的 append_token + copy-on-write，CPU/GPU overhead 都很明显。这版把 COW clone 也塞回 batch 里：先批量分配新 block，再用 index_copy_ 批量拷贝整块，最后继续走 batched kv-append。"
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

prefix cache 开起来以后，decode 入口会出现一个很典型的形态：

- 每个 request 的 KV block table 大部分都能复用（refcount>1）
- 但一旦要写入新 token，最后一个 block 需要 **copy-on-write（COW）clone**，否则会污染 cache 里的共享 block

问题在于：我们虽然已经有 `KVBlockManager.append_token_batch()`，但它的 fast path 只覆盖 `ref==1 && len<block_size`。只要 `ref>1`，就会被扔进 slow path，逐 request 调 `append_token()`：

- 逐个分配新 block
- 逐个 copy（COW clone）
- 逐个写入新 token

这会直接把 decode 的 KV 写入阶段从 “batch” 打回 “for-loop”，CPU overhead 和 kernel launch 数都会上来。

这版就做一件事：**把 shared block 的 COW clone 也批处理掉**，让它回到 fast path。

## 代码变更

### `roseinfer/engine.py`

核心思路：

1) 在 `append_token_batch()` 的 batch 扫描阶段，把 `ref>1 && len<block_size` 的项识别出来
2) 先批量分配新 block，并把 `block_ids_list[b][-1]` 改成新 gid（同时 old gid refcount--）
3) 用 `index_select + index_copy_` 批量把旧 block 拷到新 block（拷整块，保证 contiguous）
4) 这些项就可以和原本 `ref==1` 的项一起走原来的 batched kv-append（Triton/assign 都能复用）

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
     def append_token_batch(...):
@@
         fast_block_idx: list[int] = []
         fast_pos: list[int] = []
         fast_last_gid: list[int] = []
+        cow_old_block_idx: list[int] = []
+        cow_new_block_idx: list[int] = []
         slow_batch_idx: list[int] = []
@@
             info = self._block_infos[last_gid]
             ref = self._block_refcounts.get(last_gid, 1)
-            if ref != 1 or info.length >= self.block_size:
+            if info.length >= self.block_size:
                 slow_batch_idx.append(b)
                 continue
+            if ref != 1:
+                self._block_refcounts[last_gid] = ref - 1
+                block_idx = self._alloc_block_index(layer_idx)
+                new_gid = self._to_global_block_id(layer_idx, block_idx)
+                self._block_infos[new_gid] = KVBlockInfo(
+                    layer=info.layer,
+                    block_index=block_idx,
+                    start=info.start,
+                    length=info.length,
+                )
+                self._block_refcounts[new_gid] = 1
+                block_ids[-1] = new_gid
+                cow_old_block_idx.append(info.block_index)
+                cow_new_block_idx.append(block_idx)
+                fast_batch_idx.append(b)
+                fast_block_idx.append(block_idx)
+                fast_pos.append(info.length)
+                fast_last_gid.append(new_gid)
+                continue
@@
         if fast_batch_idx:
             device = self.device
             k_layer = self._k_cache[layer_idx]
             v_layer = self._v_cache[layer_idx]
+            if cow_old_block_idx:
+                old_blk_t = torch.tensor(cow_old_block_idx, device=device, dtype=torch.long)
+                new_blk_t = torch.tensor(cow_new_block_idx, device=device, dtype=torch.long)
+                k_src = k_layer.index_select(0, old_blk_t)
+                v_src = v_layer.index_select(0, old_blk_t)
+                k_layer.index_copy_(0, new_blk_t, k_src)
+                v_layer.index_copy_(0, new_blk_t, v_src)
             ...
```

### `tests/test_kv_append_token_batch_cow.py`

补一个最小单测：构造 “cache(1) + 两个 session(2) 共享同一个 last block(ref=3)” 的场景，确保 `append_token_batch()`：

- 会为两个 session 都分配新 block
- old block 最终 refcount 回到 1（只剩 cache）
- 新 block 的 prefix 内容保持一致，且各自写入的新 token 不互相污染

### `scripts/bench_kv_cow_clone.py`

把 microbench 改成 batch 版本（直接测 `append_token_batch()` 的 shared-block COW）。

## 运行

```bash
pytest -q
```

```text
.................................                                        [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
33 passed, 1 warning in 2.68s
```

## Microbench（batched COW clone）

```bash
python scripts/bench_kv_cow_clone.py --old-len 1 --batch-size 16 --iters 5000
```

Before：

```text
=== kv COW clone microbench ===
old_len: 1 batch: 16 iters: 5000 layers: 12
total: 23788.06 ms
avg per COW (clone+append+free): 24.78 us
```

After：

```text
=== kv COW clone microbench ===
old_len: 1 batch: 16 iters: 5000 layers: 12
total: 6273.07 ms
avg per COW (clone+append+free): 6.53 us
```

同样 workload 下，单个 COW 的均摊时间从 **24.78us -> 6.53us**（~3.8x）。核心原因很直观：从 “for-loop + append_token” 回到了 “batch + index_copy_ + batched kv-append”。

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 故意用 `--prompt-repeats 65`（刚好跨过一个 `block_size=64`），让每个 request 的 decode 第一个 token 都会遇到 “shared last block”的 COW。

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --prompt-repeats 65 \
  --pretok --tokenize-workers 0 \
  --num-requests 512 --max-new-tokens 2 \
  --submit-interval-ms 0 \
  --max-batch-size 16 --prefill-max-batch-size 16 \
  --prefill-admission-policy fifo \
  --paged-attn --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 1
```

Before：

```text
=== warmup 1/1 ===
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
Requests: 512
Prompt tokens (total): 33280
Completion tokens (total): 1024
Submit wall: 0.033340 s
add_request latency p50/p95/p99: 0.01/0.02/0.03 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 173.70/327.14/337.35 ms
Prefill->first token p50/p95/p99: 1.02/1.72/1.78 ms
TTFT p50/p95/p99: 174.75/328.07/338.48 ms
TPOT p50/p95/p99: 9.37/13.62/18.62 ms/token
ITL p50/p95/p99: 9.37/13.62/18.62 ms
Latency p50/p95/p99: 184.47/338.18/347.75 ms
Throughput (completion,total): 2689.83 tokens/s
```

After：

```text
=== warmup 1/1 ===
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
Requests: 512
Prompt tokens (total): 33280
Completion tokens (total): 1024
Submit wall: 0.030678 s
add_request latency p50/p95/p99: 0.01/0.02/0.03 ms
Tokenize p50/p95/p99: 0.00/0.00/0.00 ms
Queue wait (post-tok) p50/p95/p99: 130.30/225.26/232.08 ms
Prefill->first token p50/p95/p99: 0.97/1.62/1.85 ms
TTFT p50/p95/p99: 131.08/226.40/232.94 ms
TPOT p50/p95/p99: 5.85/10.06/16.55 ms/token
ITL p50/p95/p99: 5.85/10.06/16.55 ms
Latency p50/p95/p99: 138.15/232.78/238.98 ms
Throughput (completion,total): 3803.22 tokens/s
```

这组数字基本把结论说明白了：

- `ITL p50`：9.37ms -> 5.85ms
- `TTFT p50`：174.75ms -> 131.08ms
- `Throughput`：2689.83 -> 3803.22 tokens/s

shared-block 的 COW 以前会把整个 batch 拆散（slow path），现在它仍然是 batch：clone + copy + append 都能向量化，CPU overhead 明显下降，queue wait 也跟着掉下来。

## 结论

- prefix cache + paged attention 的组合里，**KV append 的 batch 语义非常关键**：一旦退化成 per-request loop，延迟/吞吐都会被拉垮。
- 这版把 `ref>1` 的 COW clone 也塞回 `append_token_batch()`，decode 的 KV 写入阶段不再“掉速”。
- 下一步如果继续往这个方向走：可以把 “需要新 block（len==block_size）” 这类分支也 batch 化；以及进一步把 block table / slot mapping 的 CPU 构造成本往下压。

