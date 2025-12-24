---
classes: wide2
title: "从零实现 LLM Inference：046. Streaming Pretok（绕过 tokenizer.encode）"
excerpt: "streaming bench 里 add_request 的 p99 很容易被 tokenizer.encode 的 CPU 开销污染。增加 add_request(prompt_token_ids=...) + benchmark_streaming --pretok，把 tokenization 从提交路径挪出去。"
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

上一版我们把 TTFT 拆成了 `Queue wait + Prefill->first token`，现在问题更明确了：**很多时候 TTFT 根本不是 decode 的问题，而是 admission/排队 + prefill**。

但在做 streaming 压测的时候，还有一个很烦的噪声源：**`add_request()` 的 CPU 开销**。

目前 `SchedulerManager.add_request(prompt=...)` 内部会 `tokenizer.encode(prompt)`，当 `num_requests` 很大时：

- `Submit wall`（把请求灌进系统需要多久）会被 tokenization 直接拖慢
- `add_request latency p99` 也会被 encode 污染，导致你很难判断调度/队列本身到底在不在抖

这一版做两件事：

1. `SchedulerManager.add_request()` 支持直接传 `prompt_token_ids`，绕过 `tokenizer.encode`
2. `benchmark_streaming.py` 增加 `--pretok`：先把 prompts 预先 tokenize 一次，然后复用 token ids

目标不是“让 TTFT 变快”，而是：**把 tokenization 从提交路径挪出去，让 streaming benchmark 更干净、更可解释**（以及给后面做 tokenization 线程池/PD 分离留 API 口子）。

## 代码变更

### `roseinfer/server.py`

`SchedulerManager.add_request()` 新增一个可选参数：

- `prompt_token_ids: Optional[list[int]] = None`

行为：

- 如果传了 `prompt_token_ids`：直接 `list(prompt_token_ids)` 作为 prompt ids（不碰 tokenizer）
- 如果 ids 为空：fallback 到 `eos_token_id`（避免空 prompt 导致 detok/session 状态错乱）
- 如果 ids 太长：截断到 `max_position_embeddings`（避免后面再因为 context length 不一致出问题）

核心 diff：

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
@@
 class SchedulerManager:
     def add_request(
         self,
         prompt: str,
+        prompt_token_ids: Optional[list[int]] = None,
         max_new_tokens: int = 64,
@@
-        token_ids = self.engine.tokenizer.encode(prompt, add_special_tokens=False)
+        if prompt_token_ids is None:
+            token_ids = self.engine.tokenizer.encode(prompt, add_special_tokens=False)
+        else:
+            token_ids = list(prompt_token_ids)
         if not token_ids:
             token_ids = [self.engine.eos_token_id]
+        max_pos = int(self.engine.config.max_position_embeddings)
+        if len(token_ids) > max_pos:
+            token_ids = token_ids[-max_pos:]
```

### `roseinfer/benchmark_streaming.py`

新增 `--pretok`：

- main 里先把 prompts `tokenizer.encode(..., add_special_tokens=False)` 一次
- 计算 `prompt_lens` 时直接用 `len(ids)`（避免重复 tokenize）
- `run_once()` 提交请求时，把 `prompt_token_ids` 一起传给 `add_request()`
- 输出里打印 `Pretok: True/False`，保证每份 benchmark 自解释

核心 diff：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
@@
     parser.add_argument(
+        "--pretok",
+        action="store_true",
+        help="Pre-tokenize prompts and pass prompt_token_ids to SchedulerManager.add_request().",
+    )
@@
 def run_once(
     *,
     engine: InferenceEngine,
     prompts: list[str],
     prompt_lens: list[int],
+    prompt_token_ids_list: list[list[int]] | None,
@@
-        for p in prompts:
+        for i, p in enumerate(prompts):
+            prompt_token_ids = (
+                prompt_token_ids_list[i] if prompt_token_ids_list is not None else None
+            )
             request_id = mgr.add_request(
                 prompt=p,
+                prompt_token_ids=prompt_token_ids,
@@
         print(f"Model: {args.hf_model_id}")
         print(f"Device: {args.device}")
+        print(f"Pretok: {bool(args.pretok)}")
@@
 def main() -> None:
@@
+    prompt_token_ids_list: list[list[int]] | None = None
+    if args.pretok:
+        ...
+        prompt_token_ids_list.append(ids)
+        prompt_lens = [len(ids) for ids in prompt_token_ids_list]
+    else:
+        prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
```

### `tests/test_server_pretok.py`

加一个最直接的语义测试：传了 `prompt_token_ids` 就不应该再走 `tokenizer.encode()`。

```diff
diff --git a/tests/test_server_pretok.py b/tests/test_server_pretok.py
@@
+def test_server_add_request_accepts_prompt_token_ids() -> None:
+    ...
+    _ = mgr.add_request(
+        \"hello\",
+        prompt_token_ids=[1, 2, 3],
+        max_new_tokens=1,
+        stop_on_eos=False,
+    )
+    assert tok.encode_calls == 0
```

## 运行

```bash
pytest -q
```

```text
.........................                                                [100%]
=============================== warnings summary ===============================
../anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260
  /data/projects/anaconda3/lib/python3.10/site-packages/urllib3/util/ssl_.py:260: DeprecationWarning: ssl.PROTOCOL_TLS is deprecated
    context = SSLContext(ssl_version or PROTOCOL_TLS)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
25 passed, 1 warning in 2.09s
```

## Benchmark（HF GPT-2 / streaming）

这版的收益点在 **提交阶段**，所以我把 `max_new_tokens` 设成 `1`，让 decode 尽量轻，专注看：

- `Submit wall`
- `add_request latency p99`

命令（不 pretok）：

```bash
python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 --device cuda \
  --prompt "Hello" --prompt-repeats 256 --unique-prompts \
  --num-requests 256 --max-new-tokens 1 \
  --max-batch-size 16 --prefill-max-batch-size 16 \
  --submit-interval-ms 0 \
  --no-stop-on-eos --no-prefix-cache \
  --warmup-runs 1 --repeat-runs 1
```

### Before（`--pretok`=off）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: False
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.077673 s
add_request latency p50/p95/p99: 0.24/0.28/0.35 ms
Queue wait p50/p95/p99: 462.51/997.02/1000.07 ms
Prefill->first token p50/p95/p99: 69.81/84.50/84.97 ms
TTFT p50/p95/p99: 533.54/1061.61/1064.58 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 534.63/1062.25/1065.11 ms
Throughput (completion,total): 224.61 tokens/s
```

命令（pretok）：

```bash
... --pretok
```

### After（`--pretok`=on）

```text
=== warmup 1/1 ===
=== streaming benchmark ===
Model: gpt2
Device: cuda
Pretok: True
Paged attention: False
CUDA Graph: False
NVTX: False
Requests: 256
Prompt tokens (total): 66304
Completion tokens (total): 256
Submit wall: 0.023336 s
add_request latency p50/p95/p99: 0.02/0.02/0.05 ms
Queue wait p50/p95/p99: 505.52/1053.60/1054.52 ms
Prefill->first token p50/p95/p99: 69.48/70.47/70.68 ms
TTFT p50/p95/p99: 575.73/1118.76/1119.61 ms
TPOT p50/p95/p99: 0.00/0.00/0.00 ms/token
ITL p50/p95/p99: 0.00/0.00/0.00 ms
Latency p50/p95/p99: 576.79/1119.08/1120.13 ms
Throughput (completion,total): 223.94 tokens/s
```

## 结论

这次的改动非常“单点”，收益也很单点，但很扎实：

- `Submit wall`：`77.7ms -> 23.3ms`（约 `3.3x`）
- `add_request latency p99`：`0.35ms -> 0.05ms`（约 `7x`）

同时也能看到：这组配置下 TTFT 基本被 `Queue wait` 主导（`~1s`），`--pretok` 不会让 TTFT 立刻变好，属于预期现象。

换句话说：**这一版不是在优化算子/调度，而是在把 benchmark 的噪声源剥离掉**。后面无论你是继续抠 admission policy，还是开始把 tokenization 挪到别的线程/进程，这个 `prompt_token_ids` 的接口都是必经的一步。
