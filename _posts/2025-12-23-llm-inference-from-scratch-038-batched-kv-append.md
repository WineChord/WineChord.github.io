---
classes: wide2
title: "从零实现 LLM Inference：038. Batched KV Append Fast Path"
excerpt: "paged decode 还在慢？很多时候瓶颈不在 attention，而是在每步 L*B 次的 KV 写入：做一个 batch fast-path，直接把 Python 循环砍掉。"
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

上一版把 `block_table` 变成 slot 常驻 GPU 以后，paged decode 的一大段 CPU overhead 被打掉了。

接下来最扎眼的就是 `roseinfer.kv.append_token`：每个 decode step 里，对每一层、每一个 request 都要跑一遍 `append_token()`。

对于 GPT-2 这种小模型（`L=12`）+ `B=64`，这意味着每个 step 需要 `12*64=768` 次 Python 调用；`512` 个 token 就是 `~39` 万次。

而绝大部分 token 的写入其实是“同一个 block 往后填一格”（refcount=1 且还没到 block 边界），完全可以走 batch fast-path：

- 一次性把这批 request 的 `k/v` 写进对应的 KV block
- 只有碰到 block 边界 / COW 才回退到原来的慢路径

## 代码变更

### `roseinfer/engine.py`

1. `KVBlockManager` 新增 `append_token_batch()`：
   - fast-path：`ref==1 && length < block_size`
   - slow-path：空 block / block 边界 / COW → 继续调用旧的 `append_token()`
2. `decode_step_sessions()` 两条路径都改成按 layer 调一次 `append_token_batch()`（从 `L*B` 次函数调用变成 `L` 次）。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
@@
-                        for idx, sess in enumerate(sessions):
-                            kvm.append_token(
-                                layer_idx,
-                                sess.block_ids_per_layer[layer_idx],
-                                k_step[idx],
-                                v_step[idx],
-                            )
+                        block_ids_list = [
+                            sess.block_ids_per_layer[layer_idx] for sess in sessions
+                        ]
+                        kvm.append_token_batch(
+                            layer_idx,
+                            block_ids_list,
+                            k_step,
+                            v_step,
+                        )
@@
+    def append_token_batch(
+        self,
+        layer_idx: int,
+        block_ids_list: list[list[int]],
+        key_new: torch.Tensor,  # [B, H, D]
+        value_new: torch.Tensor,  # [B, H, D]
+    ) -> None:
+        ...
+        if fast_batch_idx:
+            ...
+            k_layer[blk_t, :, pos_t, :] = k_src
+            v_layer[blk_t, :, pos_t, :] = v_src
+            ...
+        for b in slow_batch_idx:
+            self.append_token(layer_idx, block_ids_list[b], key_new[b], value_new[b])
```

## 运行

```bash
pytest -q
```

```text
22 passed, 1 warning in 1.64s
```

## Benchmark（HF GPT-2）

命令保持不变（把变量尽量收敛到 decode path）：

```bash
python -m rosellm.roseinfer.benchmark_scheduler \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --num-requests 64 --max-new-tokens 512 \
  --mode online --max-batch-size 64 \
  --no-stop-on-eos --no-prefix-cache --pretok \
  --warmup-runs 1 --repeat-runs 3 \
  --paged-attn
```

### Before

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 7.643991/7.639430 s
Total time p50/mean: 7.817085/7.813045 s
Throughput(completion,decode) p50/mean: 4286.77/4289.33 tokens/s
Throughput(completion,total) p50/mean: 4191.84/4194.01 tokens/s
TTFT p50/mean: 2.68/2.70 ms
TPOT p50/mean: 14.99/14.98 ms/token
Latency p50/mean: 7661.28/7659.60 ms
```

### After

```text
=== online summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 3.303571/3.302174 s
Total time p50/mean: 3.482260/3.479232 s
Throughput(completion,decode) p50/mean: 9918.96/9923.19 tokens/s
Throughput(completion,total) p50/mean: 9409.98/9418.19 tokens/s
TTFT p50/mean: 2.76/2.75 ms
TPOT p50/mean: 6.51/6.50 ms/token
Latency p50/mean: 3327.01/3325.59 ms
```

## 结论

- decode 吞吐：`4289.33 → 9923.19 tokens/s`（约 **2.31x**）
- TPOT：`14.98 → 6.50 ms/token`（约 **-56.6%**）
- 这类“离谱提升”不是因为算子突然变快了，而是把 **每步 `L*B` 次 Python 调用** 砍成了 **每步 `L` 次**，并且把 KV 写入做成 batch 写。
