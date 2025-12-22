---
classes: wide2
title: "从零实现 LLM Inference：030. Prefill Admission Knob"
excerpt: "把 worker 的 prefill admission batch size 从 decode batch size 解耦：TTFT p99 直接砍掉一半。"
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

上一版（Prefix Cache + Prefill Micro-Batching）已经做到：burst 进来时，prefill 会合成一次 batched forward。

但我很快发现一个“隐形耦合”：

- worker 每轮从 pending queue 里 **最多拿 `max_batch_size` 个 request** 去做 admission/prefill
- 这个 `max_batch_size` 同时也是 **decode step 的 batch size**

于是就出现一个很尴尬的现象：我明明只想让 **prefill admission** 更激进一点（别让后面的请求等好几轮），却不得不去动 decode batch size。

这次 mini PR 就做一件事：把这两个 knob 解耦出来。

## 代码变更

### `roseinfer/server.py`

给 `SchedulerManager` 新增一个 `prefill_max_batch_size`：

- `None` 表示保持旧行为（默认等于 `max_batch_size`）
- worker drain pending 时用 `prefill_max_batch_size`，不再用 `scheduler.max_batch_size`

```diff
diff --git a/rosellm/roseinfer/server.py b/rosellm/roseinfer/server.py
index b84fafa..42aedd2 100644
--- a/rosellm/roseinfer/server.py
+++ b/rosellm/roseinfer/server.py
@@ -111,12 +111,21 @@ class SchedulerManager:
         self,
         engine: InferenceEngine,
         max_batch_size: int = 8,
+        prefill_max_batch_size: Optional[int] = None,
         record_token_timestamps: bool = False,
     ) -> None:
+        if max_batch_size <= 0:
+            raise ValueError("max_batch_size must be positive")
+        if prefill_max_batch_size is None:
+            prefill_max_batch_size = max_batch_size
+        self._prefill_max_batch_size = int(prefill_max_batch_size)
+        if self._prefill_max_batch_size <= 0:
+            raise ValueError("prefill_max_batch_size must be positive")
+
         self.engine = engine
         self.scheduler = OnlineScheduler(
             engine,
-            max_batch_size=max_batch_size,
+            max_batch_size=int(max_batch_size),
         )
@@ -229,7 +238,7 @@ class SchedulerManager:
                 with self._lock:
                     if not self._running:
                         break
-                    max_new = self.scheduler.max_batch_size
+                    max_new = self._prefill_max_batch_size
                 pending: list[_PendingRequest] = []
                 for _ in range(max_new):
                     try:
                         pending.append(self._pending.get_nowait())
                     except queue.Empty:
```

### `roseinfer/benchmark_streaming.py`

benchmark 里也加一个 CLI 参数，方便直接 sweep：

```diff
diff --git a/rosellm/roseinfer/benchmark_streaming.py b/rosellm/roseinfer/benchmark_streaming.py
index 6462272..f9c7fcc 100644
--- a/rosellm/roseinfer/benchmark_streaming.py
+++ b/rosellm/roseinfer/benchmark_streaming.py
@@ -75,6 +75,15 @@ def parse_args() -> argparse.Namespace:
         default=8,
         help="Online scheduler max batch size (decode step)",
     )
+    parser.add_argument(
+        "--prefill-max-batch-size",
+        type=int,
+        default=None,
+        help=(
+            "Max pending requests to prefill per worker iteration "
+            "(default: same as --max-batch-size)."
+        ),
+    )
@@ -182,6 +191,7 @@ def main() -> None:
     mgr = SchedulerManager(
         engine,
         max_batch_size=int(args.max_batch_size),
+        prefill_max_batch_size=args.prefill_max_batch_size,
         record_token_timestamps=True,
     )
```

### 新增测试

主要验证两件事：

- `prefill_max_batch_size` 非法值直接 fail-fast
- `prefill_max_batch_size` 可以和 decode 的 `max_batch_size` 解耦

```diff
diff --git a/tests/test_scheduler_manager_prefill_max_batch_size.py b/tests/test_scheduler_manager_prefill_max_batch_size.py
new file mode 100644
index 0000000..2f49c13
--- /dev/null
+++ b/tests/test_scheduler_manager_prefill_max_batch_size.py
@@ -0,0 +1,65 @@
+import pytest
+import torch
+
+from rosellm.roseinfer.engine import InferenceEngine
+from rosellm.roseinfer.server import SchedulerManager
+from rosellm.rosetrainer.config import GPTConfig
+from rosellm.rosetrainer.model import GPTModel
+
+class _DummyTokenizer:
+    def __init__(self, vocab_size: int = 128) -> None:
+        self.vocab_size = int(vocab_size)
+        self.eos_token_id = 0
+        self.pad_token_id = 0
+
+    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
+        del text, add_special_tokens
+        return [1, 2, 3]
+
+    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
+        del ids, skip_special_tokens
+        return ""
+
+
+def _make_engine() -> InferenceEngine:
+    cfg = GPTConfig(
+        vocab_size=128,
+        max_position_embeddings=32,
+        n_layers=2,
+        n_heads=2,
+        d_model=32,
+        d_ff=64,
+        dropout=0.0,
+    )
+    tok = _DummyTokenizer(vocab_size=128)
+    model = GPTModel(cfg)
+    return InferenceEngine(
+        model=model,
+        config=cfg,
+        tokenizer=tok,
+        tokenizer_name="dummy",
+        device="cpu",
+        use_amp=False,
+        kv_cache_max_concurrency=4,
+        prefix_cache_max_entries=0,
+    )
+
+def test_scheduler_manager_prefill_max_batch_size_validates_positive() -> None:
+    torch.manual_seed(0)
+    engine = _make_engine()
+    with pytest.raises(ValueError, match="prefill_max_batch_size must be positive"):
+        SchedulerManager(engine, max_batch_size=2, prefill_max_batch_size=0)
+
+
+def test_scheduler_manager_prefill_max_batch_size_decouples_from_decode() -> None:
+    torch.manual_seed(0)
+    engine = _make_engine()
+    mgr = SchedulerManager(engine, max_batch_size=2, prefill_max_batch_size=8)
+    try:
+        assert mgr.scheduler.max_batch_size == 2
+        assert mgr._prefill_max_batch_size == 8
+    finally:
+        mgr.close()
```

## 指标口径

延续上一版的定义：

- TTFT：`t_first_token - t_submit`
- TPOT：`(t_last_token - t_first_token) / (n_tokens - 1)`
- ITL：`t_i - t_{i-1}`（所有 token 的间隔摊平看分布）

这版关注点很明确：**prefill admission 是否会把 tail TTFT 拉爆**。

## 运行

### 单测

```shell
pytest -q
```

输出：

```text
...........                                                              [100%]
11 passed, 1 warning in 1.61s
```

### Benchmark（HF GPT-2）

命令：

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --unique-prompts \
  --num-requests 32 \
  --max-new-tokens 8 \
  --no-stop-on-eos \
  --max-batch-size 8
```

#### Before（prefill admission == decode batch size）

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.082981 s
add_request latency p50/p95/p99: 0.03/0.11/53.13 ms
TTFT p50/p95/p99: 140.34/255.57/255.68 ms
TPOT p50/p95/p99: 105.36/107.14/110.48 ms/token
ITL p50/p95/p99: 111.87/128.46/154.48 ms
Latency p50/p95/p99: 891.26/957.18/957.38 ms
Throughput (completion,total): 246.20 tokens/s
```

#### After（prefill admission 独立 knob：一次吞掉更多 pending）

只改一个参数：`--prefill-max-batch-size 32`（decode 仍然是 `--max-batch-size 8`）。

```shell
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rosellm.roseinfer.benchmark_streaming \
  --hf-model-id gpt2 \
  --device cpu \
  --prompt "Hello" \
  --unique-prompts \
  --num-requests 32 \
  --max-new-tokens 8 \
  --no-stop-on-eos \
  --max-batch-size 8 \
  --prefill-max-batch-size 32
```

```text
=== streaming benchmark ===
Model: gpt2
Device: cpu
Requests: 32
Prompt tokens (total): 128
Completion tokens (total): 256
Submit wall: 0.088231 s
add_request latency p50/p95/p99: 0.04/0.17/57.27 ms
TTFT p50/p95/p99: 115.74/117.66/117.99 ms
TPOT p50/p95/p99: 101.73/109.35/109.35 ms/token
ITL p50/p95/p99: 103.90/131.18/141.62 ms
Latency p50/p95/p99: 828.82/880.04/880.14 ms
Throughput (completion,total): 264.57 tokens/s
```

核心收益看 tail 最直观：

- TTFT p99：`255.68ms -> 117.99ms`（~2.2x）
- Latency p99：`957.38ms -> 880.14ms`（~1.09x）
- TPOT/ITL：基本不变（因为 decode batch size 没动）
