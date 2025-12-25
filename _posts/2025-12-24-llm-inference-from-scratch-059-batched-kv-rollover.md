---
classes: wide2
title: "从零实现 LLM Inference：059. Batched KV rollover（block 满了也别退化成 for-loop）"
excerpt: "append_token_batch 之前只覆盖 len<block_size 的 fast path；一旦 last block 满了（len==block_size）就会退化成逐 request 的 append_token。这个点会制造 ITL 的尖刺。这版把 rollover 也塞回 batch：先批量分配新 block，再继续走 batched kv-append。"
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

KV cache 用 block 来组织以后，每个 request 的长度每增长 `block_size` 次，就会遇到一次 “rollover”：

- last block 写满（`len==block_size`）
- 下一 token 需要新 block

如果这一步做得不好，会出现一个很典型的现象：**ITL 里每隔 64 token 就有一次尖刺**（GPU 也容易出现空洞）。

之前 `KVBlockManager.append_token_batch()` 的 fast path 只覆盖 `len<block_size && ref==1`。一旦 block 满了，就会被扔进 slow path，逐 request 调 `append_token()`，把 batch 打回 for-loop。

这版做的事情很小：**在 batch 扫描阶段把 “len==block_size” 的项提前 rollover 掉**，让它们继续留在 fast path。

## 代码变更

### `roseinfer/engine.py`

思路：

1) 扫描 `block_ids_list` 时，如果发现 `len==block_size`：
   - 直接分配一个新 block（`start += block_size, length=0`）
   - `block_ids.append(new_gid)`
2) rollover 完以后，所有项都能统一走原本的 batched kv-append

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
         for b, block_ids in enumerate(block_ids_list):
-            if not block_ids:
-                slow_batch_idx.append(b)
-                continue
-            last_gid = block_ids[-1]
-            info = self._block_infos[last_gid]
-            ref = self._block_refcounts.get(last_gid, 1)
-            if info.length >= self.block_size:
-                slow_batch_idx.append(b)
-                continue
+            if not block_ids:
+                ...  # 先分配首 block（可选，但顺手补齐）
+            else:
+                ...
+
+            if info.length >= self.block_size:
+                block_idx = self._alloc_block_index(layer_idx)
+                new_gid = self._to_global_block_id(layer_idx, block_idx)
+                info = KVBlockInfo(
+                    layer=info.layer,
+                    block_index=block_idx,
+                    start=info.start + info.length,
+                    length=0,
+                )
+                self._block_infos[new_gid] = info
+                self._block_refcounts[new_gid] = 1
+                block_ids.append(new_gid)
+                last_gid = new_gid
+                ref = 1
```

### `tests/test_kv_append_token_batch_rollover.py`

补一个最小单测：两条 session 共享一个“满 block（ref=2）”，batch append 时：

- 不做 COW（因为旧 block 不再被写）
- 为每条 session 都 append 一个新 block，并把 token 写进新 block 的 pos=0

## 运行

```bash
pytest -q
```

```text
..................................                                       [100%]
34 passed, 1 warning in 2.68s
```

## Benchmark（HF GPT-2 / streaming）

这组 benchmark 关闭 prefix cache（避免其它因素），并把 `--max-new-tokens 256` 拉长，让 rollover 发生多次（每个 request 大概会遇到 4 次边界）。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt 'Hello' --pretok --tokenize-workers 0 \
  --num-requests 256 --max-new-tokens 256 \
  --submit-interval-ms 0 \
  --max-batch-size 16 --prefill-max-batch-size 16 \
  --prefill-admission-policy fifo \
  --paged-attn --no-prefix-cache --no-stop-on-eos \
  --warmup-runs 1 --repeat-runs 1
```

Before：

```text
TPOT p50/p95/p99: 69.36/69.45/69.48 ms/token
ITL p50/p95/p99: 67.97/78.54/95.34 ms
Throughput (completion,total): 3665.72 tokens/s
```

After：

```text
TPOT p50/p95/p99: 67.01/67.08/67.11 ms/token
ITL p50/p95/p99: 66.39/70.99/81.67 ms
Throughput (completion,total): 3791.82 tokens/s
```

这里最重要的是 **ITL p99 的尖刺明显收敛**：95.34ms -> 81.67ms。因为 “边界那一步” 不再退化成逐 request 的 append_token。

## 结论

- 只要系统里还存在 “偶发退化成 for-loop” 的路径，就会在 tail latency 上留下印记（ITL/TPOT 的 p99 最直观）。
- rollover 是 KV block 组织的必经之路，必须把它放回 batch fast path。
- 下一步如果继续抠 KV 写入：可以继续把 “更多 slow path 分支” 拉回 batch（比如其它 allocate/copy 分支），并进一步减少每 step 的 Python 构造开销。

