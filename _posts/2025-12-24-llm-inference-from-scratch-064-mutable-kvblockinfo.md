---
classes: wide2
title: "从零实现 LLM Inference：064. KVBlockInfo 改成可变（减少 per-token KV 元数据开销）"
excerpt: "KV append 的热路径里，每层每 token 都要更新一次 block length。之前用 NamedTuple 需要不断创建新对象并回写 dict；这版改成 slots dataclass，length 原地自增，减少 Python 分配和重复查表。"
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

`KVBlockManager.append_token_batch()` 是 decode 的热路径之一：每个 decode step、每一层都要把新 token 的 KV 写进 cache，同时更新 “这个 block 现在写到哪了（length/pos）”。

之前 `KVBlockInfo` 用的是 `NamedTuple`（不可变），所以每次 length+1 都得：

- 重新构造一个 `KVBlockInfo(...)`
- 写回 `self._block_infos[gid]`

这在小模型（GPT-2）+ 高并发时会比较容易变成 CPU hot loop。这里把它改成 **slots dataclass + 原地自增**，把这段纯 Python 的对象/字典开销压下去。

## 代码变更

### `roseinfer/engine.py`

核心 diff：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
-class KVBlockInfo(NamedTuple):
+@dataclass(slots=True)
+class KVBlockInfo:
     layer: int
     block_index: int
     start: int
     length: int
@@
-new_info = KVBlockInfo(..., length=info.length + 1)
-self._block_infos[last_id] = new_info
+info.length += 1
@@
-fast_last_gid: list[int] = []
@@
-fast_pos.append(info.length)
-fast_last_gid.append(last_gid)
+fast_pos.append(info.length)
+info.length += 1
@@
-for gid in fast_last_gid:
-    info = self._block_infos[gid]
-    self._block_infos[gid] = KVBlockInfo(..., length=info.length + 1)
 ```

## 运行

```bash
pytest -q
```

```text
.....................................                                    [100%]
37 passed, 1 warning in 2.73s
```

## Benchmark（HF GPT-2 / streaming / decode-heavy）

这组 benchmark 主要想把 `append_token_batch()` 的调用次数拉满：

- `--no-prefix-cache`：避免 prefix cache 相关分支干扰
- `--max-new-tokens 128`：让每个 request 跑足够多的 decode steps
- `--prompt-repeats "1,4"`：让 batch 里 prompt 长短混合（pos 不恒定）

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
Throughput: 2091.74 / 2110.63 / 2098.34 tokens/s (avg 2100.24)
TPOT p99: 60.70 / 60.14 / 60.52 ms/token (avg 60.45)
ITL p99: 75.37 / 68.54 / 75.79 ms (avg 73.23)
```

After（3 次 run）：

```text
Throughput: 2127.74 / 2181.76 / 2156.18 tokens/s (avg 2155.23)
TPOT p99: 59.58 / 58.16 / 58.86 ms/token (avg 58.87)
ITL p99: 78.40 / 63.05 / 66.48 ms (avg 69.31)
```

## 结论

- completion throughput 平均 **+2.62%**（`2100.24 -> 2155.23 tokens/s`）
- `TPOT p99` 平均 **-2.62%**（`60.45 -> 58.87 ms/token`）
- `ITL p99` 平均 **-5.36%**（`73.23 -> 69.31 ms`）

这类改动不改变算子/内存访问模式，纯粹是在 hot loop 里减少 Python 对象和重复查表：对 **小模型 / 高并发 / decode steps 多** 的场景更敏感。

