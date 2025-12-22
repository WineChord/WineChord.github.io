---
classes: wide2
title: "从零实现 LLM Inference：028. Prefill Micro-Batching"
excerpt: "把 worker 里的 prefill 从串行变成 micro-batch，一次 forward 吞掉一批 pending request。"
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

上一版（Pending Queue Admission）把 streaming 的 `add_request()` 做到非常快：请求几乎立刻入队。

但新问题也很直观：**prefill 仍然在 worker 里按 request 串行跑**。burst 一来，TTFT/尾延迟会被这一段“串行 prefill 队列”直接拉爆。

这次 mini PR 就做一件事：**prefill micro-batching** —— 每轮 worker 从 pending 里拿一批 request，合成一次 batched prefill forward，再把 KV 写进 block manager，然后逐个 sample 第一个 token。

## Benchmark（HF GPT-2）

命令（同一个命令，直接对比）：

```shell
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --unique-prompts \
  --num-requests 32 \
  --max-new-tokens 8 \
  --no-stop-on-eos \
  --no-prefix-cache
```

### Before（prefill 串行）

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.091374 s
add_request latency p50/p95/p99: 0.07/0.14/58.02 ms
TTFT p50/p95/p99: 405.64/744.27/770.22 ms
Latency p50/p95/p99: 1354.81/1417.39/1417.53 ms
Throughput (completion,total): 169.79 tokens/s
```

### After（prefill micro-batch）

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.082430 s
add_request latency p50/p95/p99: 0.04/0.12/52.68 ms
TTFT p50/p95/p99: 132.30/246.50/246.68 ms
Latency p50/p95/p99: 840.31/907.19/907.29 ms
Throughput (completion,total): 258.81 tokens/s
```

这几个数字基本就够说明问题：

- TTFT p50：`405.64ms -> 132.30ms`（~3.1x）
- TTFT p95：`744.27ms -> 246.50ms`（~3.0x）
- Latency p50：`1354.81ms -> 840.31ms`（~1.6x）
- Throughput：`169.79 -> 258.81 tokens/s`（~1.5x）

## 代码变更

### `roseinfer/engine.py`

核心点：

1. 新增 `OnlineRequest` + `OnlineScheduler.add_requests()`，支持一次性 admission 多个 request。
2. `InferenceEngine` 新增 `_encode_prompt_token_ids_batch()`（left-pad + attention_mask）。
3. 新增 `_prefill_register_kv_batch()`：一次 forward 得到 `presents`，按 session 的真实长度 slice，再逐条 `register_prefill_layer()` 写进 KV block manager。
4. 这版先保持 PR 足够小：**prefix cache 开启时直接 fallback 单条 add_request**，后面再做融合。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 34e2370..4351307 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -1,4 +1,5 @@
 from collections import OrderedDict, deque
+from dataclasses import dataclass
 from typing import Iterator, NamedTuple, Optional
@@ -145,6 +146,49 @@ class InferenceEngine:
     def _encode_prompt_token_ids(self, token_ids: list[int]) -> torch.Tensor:
         ids = list(token_ids)
         if not ids:
             ids = [self.eos_token_id]
         input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
         return input_ids  # [1, T0]
+
+    def _encode_prompt_token_ids_batch(
+        self,
+        token_ids_list: list[list[int]],
+    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[int]]]:
+        if not token_ids_list:
+            raise ValueError("token_ids_list must be non-empty")
+        max_pos = int(self.config.max_position_embeddings)
+        pad_id = self.tokenizer.pad_token_id
+        if pad_id is None:
+            pad_id = self.eos_token_id
+
+        truncated: list[list[int]] = []
+        lengths: list[int] = []
+        max_len = 0
+        for ids0 in token_ids_list:
+            ids = list(ids0)
+            if not ids:
+                ids = [self.eos_token_id]
+            if len(ids) > max_pos:
+                ids = ids[-max_pos:]
+            truncated.append(ids)
+            lengths.append(len(ids))
+            max_len = max(max_len, len(ids))
+
+        batch: list[list[int]] = []
+        masks: list[list[int]] = []
+        for ids in truncated:
+            pad_len = max_len - len(ids)
+            batch.append([pad_id] * pad_len + ids)
+            masks.append([0] * pad_len + [1] * len(ids))
+
+        input_ids = torch.tensor(
+            batch,
+            dtype=torch.long,
+            device=self.device,
+        )
+        attention_mask = torch.tensor(
+            masks,
+            dtype=torch.long,
+            device=self.device,
+        )
+        return input_ids, attention_mask, lengths, truncated
@@ -265,6 +309,59 @@ class InferenceEngine:
         if max_new_tokens > 0 and session.step_count >= max_new_tokens:
             session.finished = True
+
+    @torch.no_grad()
+    def _prefill_register_kv_batch(
+        self,
+        sessions: list["InferenceSession"],
+        input_ids: torch.Tensor,  # [B, T]
+        attention_mask: torch.Tensor,  # [B, T]
+        lengths: list[int],  # [B]
+    ) -> torch.Tensor:
+        if len(sessions) != input_ids.size(0) or len(lengths) != input_ids.size(0):
+            raise ValueError("batch size mismatch")
+        from torch.amp import autocast
+
+        position_ids = attention_mask.to(dtype=torch.long).cumsum(-1) - 1
+        position_ids.masked_fill_(attention_mask == 0, 0)
+
+        with record_function("roseinfer.prefill_batch.model_forward"):
+            if self.use_amp:
+                with autocast(
+                    device_type=self.device.type,
+                    dtype=self.amp_dtype,
+                ):
+                    logits, _, presents = self.model(
+                        input_ids=input_ids,
+                        attention_mask=attention_mask,
+                        labels=None,
+                        past_key_values=None,
+                        use_cache=True,
+                        position_ids=position_ids,
+                    )
+            else:
+                logits, _, presents = self.model(
+                    input_ids=input_ids,
+                    attention_mask=attention_mask,
+                    labels=None,
+                    past_key_values=None,
+                    use_cache=True,
+                    position_ids=position_ids,
+                )
+        kvm = self.kv_manager
+        with record_function("roseinfer.prefill_batch.register_kv"):
+            for layer_idx, layer_past in enumerate(presents):
+                if layer_idx >= kvm.num_layers:
+                    break
+                k_layer, v_layer = layer_past  # [B, H, T, D]
+                for b, sess in enumerate(sessions):
+                    seq_len = int(lengths[b])
+                    sess.prompt_length = seq_len
+                    k = k_layer[b : b + 1, :, -seq_len:, :]
+                    v = v_layer[b : b + 1, :, -seq_len:, :]
+                    block_ids = kvm.register_prefill_layer(
+                        layer_idx,
+                        k,
+                        v,
+                    )
+                    sess.block_ids_per_layer[layer_idx] = block_ids
+        last_logits = logits[:, -1, :]  # [B, V]
+        return last_logits
@@ -1393,6 +1490,19 @@ class OfflineScheduler:
         return outputs
+
+
+@dataclass(frozen=True)
+class OnlineRequest:
+    prompt: str
+    max_new_tokens: int = 64
+    temperature: float = 1.0
+    top_k: int = 0
+    top_p: float = 1.0
+    stop_on_eos: bool = True
+    do_sample: bool = False
+    prompt_token_ids: Optional[list[int]] = None
+    request_id: Optional[int] = None
@@ -1452,6 +1562,24 @@ class OnlineScheduler:
+    def add_requests(
+        self,
+        requests: list[OnlineRequest],
+    ) -> list[int]:
+        if not requests:
+            return []
+        if self.use_prefix_cache:
+            return [
+                self.add_request(
+                    prompt=req.prompt,
+                    max_new_tokens=req.max_new_tokens,
+                    temperature=req.temperature,
+                    top_k=req.top_k,
+                    top_p=req.top_p,
+                    stop_on_eos=req.stop_on_eos,
+                    do_sample=req.do_sample,
+                    prompt_token_ids=req.prompt_token_ids,
+                    request_id=req.request_id,
+                )
+                for req in requests
+            ]
+
+        # prefix cache 关闭时：batch prefill -> register kv -> per-request sample
```

### `roseinfer/server.py`

worker admission 从 “for req in pending: add_request()” 变成 “一次 add_requests()”，这样每轮最多 `max_batch_size` 条 request 只需要 **一次 prefill forward**。

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index 81f29d8..8479835 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -12,7 +12,7 @@ from fastapi.responses import StreamingResponse
 from pydantic import BaseModel
 
 from .detokenizer import BaseDetokenizer
-from .engine import InferenceEngine, OnlineScheduler
+from .engine import InferenceEngine, OnlineRequest, OnlineScheduler
@@ -222,6 +222,7 @@ class SchedulerManager:
                         pending.append(self._pending.get_nowait())
                     except queue.Empty:
                         break
+                batch: list[OnlineRequest] = []
                 for req in pending:
                     with self._lock:
                         if not self._running:
@@ -230,17 +231,21 @@ class SchedulerManager:
                         detok = self._detoks.get(req.request_id)
                     if q is None or detok is None:
                         continue
-                    rid = self.scheduler.add_request(
-                        prompt=req.prompt,
-                        max_new_tokens=req.max_new_tokens,
-                        temperature=req.temperature,
-                        top_k=req.top_k,
-                        top_p=req.top_p,
-                        stop_on_eos=req.stop_on_eos,
-                        do_sample=req.do_sample,
-                        prompt_token_ids=req.prompt_token_ids,
-                        request_id=req.request_id,
+                    batch.append(
+                        OnlineRequest(
+                            prompt=req.prompt,
+                            max_new_tokens=req.max_new_tokens,
+                            temperature=req.temperature,
+                            top_k=req.top_k,
+                            top_p=req.top_p,
+                            stop_on_eos=req.stop_on_eos,
+                            do_sample=req.do_sample,
+                            prompt_token_ids=req.prompt_token_ids,
+                            request_id=req.request_id,
+                        )
                     )
+                rids = self.scheduler.add_requests(batch) if batch else []
+                for rid in rids:
                     with self._lock:
                         q = self._queues.get(rid)
                         detok = self._detoks.get(rid)
```

## 运行

```shell
pytest -q
```

输出：

```text
.......                                                                  [100%]
7 passed, 1 warning in 1.61s
```
