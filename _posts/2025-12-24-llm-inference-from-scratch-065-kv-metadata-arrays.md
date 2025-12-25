---
classes: wide2
title: "从零实现 LLM Inference：065. KV 元数据数组化（_block_infos / _block_refcounts）"
excerpt: "KVBlockManager 的 block 元数据原来用 dict 做 global_id -> info/refcount 映射，decode 热路径会频繁查表。这里把两张表改成定长 list（按 global_id 直接索引），减少 Python dict 开销。"
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

`KVBlockManager.append_token_batch()` 是 decode 的热路径：每一层、每一步都要根据 `block_ids` 找到 “当前 last block 的 index / length”，然后写 KV 并更新 length。

之前 `KVBlockManager` 维护两张 dict：

- `_block_infos: dict[int, KVBlockInfo]`
- `_block_refcounts: dict[int, int]`

但我们这里的 `global_id = layer_idx * max_blocks_per_layer + block_idx` 本质上是 **连续整数 id**，而且上界固定（`num_layers * max_blocks_per_layer`）。

所以这版直接把两张 dict 改成 **定长 list**：按 `global_id` 直接索引，减少 decode 热路径的 Python dict 开销。

## 代码变更

### `roseinfer/engine.py`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@ class KVBlockManager.__init__
-self._block_infos: dict[int, KVBlockInfo] = {}
+self._block_infos: list[KVBlockInfo | None] = [None for _ in range(num_layers * max_blocks_per_layer)]
@@
-self._block_refcounts: dict[int, int] = {}
+self._block_refcounts: list[int] = [0 for _ in range(num_layers * max_blocks_per_layer)]
@@ def incref_blocks(...)
-self._block_refcounts[gid] = self._block_refcounts.get(gid, 0) + 1
+self._block_refcounts[gid] += 1
@@ def free_blocks(...)
-ref = self._block_refcounts.get(gid)
+ref = self._block_refcounts[gid]
 ...
-info = self._block_infos.pop(gid, None)
+info = self._block_infos[gid]; self._block_infos[gid] = None
```

## 运行

```bash
pytest -q
```

```text
.....................................                                    [100%]
37 passed, 1 warning in 2.69s
```

## Benchmark（HF GPT-2 / streaming / decode-heavy）

为了让 `append_token_batch()` 的调用次数足够多，用一组 decode-heavy 的配置：

- `--no-prefix-cache`：避免 prefix cache 分支干扰
- `--max-new-tokens 128`：让每个 request 产生足够多的 decode steps
- `--prompt-repeats "1,4"`：混合不同 prompt 长度（pos 不恒定）

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats "1,4" \
  --pretok --num-requests 128 \
  --max-new-tokens 128 --no-stop-on-eos \
  --no-prefix-cache --paged-attn \
  --warmup-runs 1 --repeat-runs 3
```

Before（3 次 run）：

```text
Throughput: 2105.17 / 2114.89 / 2114.55 tokens/s (avg 2111.54)
TPOT p99: 60.27 / 60.05 / 60.05 ms/token (avg 60.12)
```

After（3 次 run）：

```text
Throughput: 2147.02 / 2131.60 / 2114.18 tokens/s (avg 2130.93)
TPOT p99: 59.13 / 59.54 / 60.06 ms/token (avg 59.58)
```

## 结论

- completion throughput 平均 **+0.92%**（`2111.54 -> 2130.93 tokens/s`）
- `TPOT p99` 平均 **-0.91%**（`60.12 -> 59.58 ms/token`）

这类优化属于典型 “把 hot path 的 dict 查表换成数组索引”：收益不大但稳定可控，也为后续更激进的 KV 热路径优化（buffer 复用 / kernel 融合）打基础。

