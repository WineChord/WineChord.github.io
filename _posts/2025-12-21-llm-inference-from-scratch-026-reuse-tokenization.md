---
classes: wide2
title: "从零实现 LLM Inference：026. Reuse Tokenization"
excerpt: "OnlineScheduler 支持 prompt_token_ids，server 去掉重复 encode，benchmark 增加 --pretok。"
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

这一版做一个很小但很实际的优化：**prompt 的 tokenization 只做一次**。

之前在 server 侧为了 streaming detokenizer，需要先 `tokenizer.encode(prompt)` 一次；但 `OnlineScheduler.add_request(prompt)` 内部又会再 encode 一次，属于纯 CPU 重复工作。

同时 benchmark 的 online 模式里，`prefill/add` 计时区间也会把 `tokenizer.encode` 算进去，不够“纯”。

所以这次 mini PR 做三件事：

1. `OnlineScheduler.add_request()` 支持直接传 `prompt_token_ids`。
2. `server.py` 复用这份 token ids，不再二次 encode。
3. `benchmark_scheduler.py` 增加 `--pretok`，把 tokenization 移到计时区间外。

## 代码变更

### `roseinfer/engine.py`

核心点：

- `InferenceEngine` 新增 `_encode_prompt_token_ids()`。
- `_maybe_prefill_with_prefix_cache()` 新增可选参数 `prompt_token_ids`，优先走 token ids 分支。
- `OnlineScheduler.add_request()` 新增参数 `prompt_token_ids` 并透传到 engine。

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@
     def _encode_prompt(self, prompt: str) -> torch.Tensor:
         ids = self.tokenizer.encode(prompt, add_special_tokens=False)
@@
+    def _encode_prompt_token_ids(self, token_ids: list[int]) -> torch.Tensor:
+        ids = list(token_ids)
+        if not ids:
+            ids = [self.eos_token_id]
+        return torch.tensor([ids], dtype=torch.long, device=self.device)
@@
     def _maybe_prefill_with_prefix_cache(
@@
+        prompt_token_ids: Optional[list[int]] = None,
     ) -> None:
-        input_ids = self._encode_prompt(prompt)
+        if prompt_token_ids is None:
+            input_ids = self._encode_prompt(prompt)
+        else:
+            input_ids = self._encode_prompt_token_ids(prompt_token_ids)
@@
 class OnlineScheduler:
@@
     def add_request(
@@
+        prompt_token_ids: Optional[list[int]] = None,
     ) -> int:
@@
             stop_on_eos=stop_on_eos,
+            prompt_token_ids=prompt_token_ids,
         )
```

### `roseinfer/server.py`

server 里本来就已经算出了 `token_ids`，直接传下去：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@
             request_id = self.scheduler.add_request(
                 prompt=prompt,
@@
                 do_sample=do_sample,
+                prompt_token_ids=token_ids,
             )
```

### `roseinfer/benchmark_scheduler.py`

加 `--pretok` 开关；打开时提前把每个 prompt encode 好，然后传 `prompt_token_ids`：

```diff
diff --git a/rosellm/roseinfer/benchmark_scheduler.py b/rosellm/roseinfer/benchmark_scheduler.py
--- a/rosellm/roseinfer/benchmark_scheduler.py
+++ b/rosellm/roseinfer/benchmark_scheduler.py
@@
     parser.add_argument(
+        "--pretok",
+        action="store_true",
+        help="Pre-tokenize prompts outside timed region (online/offline mode).",
+    )
@@
         for i, p in enumerate(prompts):
+            prompt_token_ids = None
+            if prompt_token_ids_list is not None:
+                prompt_token_ids = prompt_token_ids_list[i]
             rid = scheduler.add_request(
                 p,
+                prompt_token_ids=prompt_token_ids,
```

### `tests/`

补了两个小测试：

- scheduler：传了 `prompt_token_ids` 就不应该再触发 tokenizer.encode
- server：`SchedulerManager.add_request()` 只 encode 一次

## 运行

```shell
pytest -q
```

输出：

```text
.....                                                                    [100%]
5 passed, 1 warning in 1.55s
```

再跑一个 online benchmark（把 tokenization 移到计时区间外）：

```shell
python -m rosellm.roseinfer.benchmark_scheduler --hf-model-id gpt2 --device cpu --prompt "Hello" --num-requests 128 --max-new-tokens 1 --mode online --warmup-runs 0 --repeat-runs 1 --pretok
```

输出：

```text
=== online summary ===
Warmup runs: 0
Measured runs: 1
Decode time p50/mean: 0.001256/0.001256 s
Total time p50/mean: 0.051682/0.051682 s
Throughput(completion,decode) p50/mean: 101950.68/101950.68 tokens/s
Throughput(completion,total) p50/mean: 2476.67/2476.67 tokens/s
TTFT p50/mean: 0.21/0.21 ms
TPOT p50/mean: 0.00/0.00 ms/token
Latency p50/mean: 0.21/0.21 ms
```
