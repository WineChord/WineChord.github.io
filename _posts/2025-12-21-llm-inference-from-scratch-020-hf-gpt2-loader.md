---
classes: wide2
title: "从零实现 LLM Inference：020. HuggingFace GPT2 Loader"
excerpt: "支持从 HuggingFace 加载 GPT2 权重，为后续和 vLLM/sglang 对齐 benchmark 铺路。"
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

在上一次把 paged attention 跑通之后，我接下来最想做的一件事是：**能和主流推理框架做一个比较公平的性能对比**。

但目前 roseinfer 的模型加载方式还是“只认我自己训练出来的 checkpoint”，这会导致两件事：

1. 我想换一个模型验证性能（比如 HF 上的 gpt2 / llama / qwen），成本很高。
2. 我想和 vLLM / sglang 做对齐 benchmark 时，很难做到 apples-to-apples。

所以这次我们先做一个很小的 mini PR：**支持从 HuggingFace 直接加载 GPT2 权重到我们自己的 `GPTModel`**，并且把 benchmark 脚本一键跑通。

这里我刻意没有直接上 Qwen3（比如 `Qwen/Qwen3-0.6B`），因为它是 LLaMA 系结构：RoPE / RMSNorm / SwiGLU / GQA… 当前我们的 `GPTModel` 还是 GPT2 风格，强行兼容会把 PR 变成“大重构”，不利于迭代。

## 对齐 GPT2 的激活函数（gelu_new）

HF 的 GPT2 默认 activation 是 `gelu_new`（tanh 近似版），而我们之前的 FFN 写死 `F.gelu`，权重即使加载对了，前向也会有数值偏差。

所以我在 `GPTConfig` 里加了一个 `activation` 字段，并在 `FeedForward` 里分支实现：

```diff
diff --git a/rosellm/rosetrainer/config.py b/rosellm/rosetrainer/config.py
index 44fa20a..de75450 100644
--- a/rosellm/rosetrainer/config.py
+++ b/rosellm/rosetrainer/config.py
@@ -10,5 +10,6 @@ class GPTConfig:
     d_model: int = 768
     d_ff: int = 3072
     dropout: float = 0.1
+    activation: str = "gelu"
     use_tensor_parallel: bool = False
     use_activation_checkpoint: bool = False
```

```diff
diff --git a/rosellm/rosetrainer/model.py b/rosellm/rosetrainer/model.py
index 9cb9c2b..5775d7a 100644
--- a/rosellm/rosetrainer/model.py
+++ b/rosellm/rosetrainer/model.py
@@ -147,6 +147,12 @@ class MultiHeadSelfAttention(nn.Module):
         return out
 
+def gelu_new(x: torch.Tensor) -> torch.Tensor:
+    return (
+        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
+    )
+
 class FeedForward(nn.Module):
@@ -170,10 +176,14 @@ class FeedForward(nn.Module):
         self.fc2.gpt2_residual = True
         self.dropout = nn.Dropout(config.dropout)
+        self.activation = getattr(config, "activation", "gelu")
 
     def forward(self, x: torch.Tensor):
         x = self.fc1(x)
-        x = F.gelu(x)
+        if self.activation == "gelu_new":
+            x = gelu_new(x)
+        else:
+            x = F.gelu(x)
         x = self.fc2(x)
         x = self.dropout(x)
         return x
```

这里的设计点是：**默认行为不变**，旧 checkpoint 的 config 没有 `activation` 字段也能继续跑；只有 HF GPT2 loader 会把 `activation` 设置成 `gelu_new`。

## HF GPT2 权重映射：Conv1D vs Linear

接下来核心就是权重映射。HF GPT2 的线性层叫 `Conv1D`（历史包袱），它的 weight 存的是 `[in, out]`，而 `nn.Linear` 的 weight 是 `[out, in]`，所以 mapping 的关键就是一个 `transpose()`。

我把这块写成了一个独立的小模块 `rosellm/rosetrainer/hf_gpt2.py`，里面主要做三件事：

1. `gpt_config_from_hf_gpt2(hf_cfg)`：把 HF 的 config 转成我们自己的 `GPTConfig`
2. `convert_hf_gpt2_state_dict(hf_sd)`：把 HF 的 state_dict 转成我们 `GPTModel` 的 key
3. `load_gpt2_from_hf_pretrained(model_id)`：直接 `from_pretrained` 然后 load 到 `GPTModel`

其中最关键的映射规则大概是：

```python
def _t(w: torch.Tensor) -> torch.Tensor:
    return w.t().contiguous()

# attention
out[f"blocks.{i}.attn.qkv_proj.weight"] = _t(hf_sd[f"transformer.h.{i}.attn.c_attn.weight"])
out[f"blocks.{i}.attn.qkv_proj.bias"] = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"]
out[f"blocks.{i}.attn.out_proj.weight"] = _t(hf_sd[f"transformer.h.{i}.attn.c_proj.weight"])
out[f"blocks.{i}.attn.out_proj.bias"] = hf_sd[f"transformer.h.{i}.attn.c_proj.bias"]

# mlp
out[f"blocks.{i}.mlp.fc1.weight"] = _t(hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"])
out[f"blocks.{i}.mlp.fc1.bias"] = hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"]
out[f"blocks.{i}.mlp.fc2.weight"] = _t(hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"])
out[f"blocks.{i}.mlp.fc2.bias"] = hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"]

# tie weights
out["token_embedding.weight"] = hf_sd["transformer.wte.weight"]
out["lm_head.weight"] = out["token_embedding.weight"]
```

这里我还踩了一个小坑：`low_cpu_mem_usage=True` 会要求安装 `accelerate`，但我这边环境默认没有，所以我做了一个 try/except：有 accelerate 才开 `low_cpu_mem_usage`，没有就降级正常加载。

## InferenceEngine：支持注入 model/config/tokenizer

之前 `InferenceEngine` 构造函数里写死了 `torch.load(checkpoint)`，这不利于后续扩展：加载 HF、加载 TensorRT engine、甚至加载别的编译产物都很难做。

所以我把它改成了依赖注入（还是保留旧路径完全兼容）：

```diff
diff --git a/rosellm/roseinfer/engine.py b/rosellm/roseinfer/engine.py
index 1e60d01..c14ba5e 100644
--- a/rosellm/roseinfer/engine.py
+++ b/rosellm/roseinfer/engine.py
@@ -22,7 +22,7 @@ except ImportError:
 class InferenceEngine:
     def __init__(
         self,
-        checkpoint_path: str,
+        checkpoint_path: str | None = None,
@@ -31,6 +31,9 @@ class InferenceEngine:
         kv_cache_max_concurrency: int = 256,
         prefix_cache_max_entries: int = 256,
         use_paged_attention: bool = False,
+        model: GPTModel | None = None,
+        config: GPTConfig | None = None,
+        tokenizer=None,
     ) -> None:
@@ -45,24 +48,54 @@ class InferenceEngine:
         else:
             self.amp_dtype = None
-        ckpt = torch.load(checkpoint_path, map_location=self.device.type)
+        if model is None:
+            if checkpoint_path is None:
+                raise ValueError("checkpoint_path must be provided when model is None")
+            ckpt = torch.load(checkpoint_path, map_location=self.device.type)
             ...
+            self.model = GPTModel(config).to(self.device)
+            self.model.load_state_dict(ckpt["model"])
         else:
+            if config is None:
+                raise ValueError("config must be provided when model is not None")
+            self.config = config
+            self.model = model.to(self.device)
         self.model.eval()
-        self.tokenizer = build_tokenizer(tokenizer_name)
+        if tokenizer is None:
+            self.tokenizer = build_tokenizer(tokenizer_name)
+        else:
+            self.tokenizer = tokenizer
```

设计点也很简单：**InferenceEngine 负责调度/缓存/服务，模型怎么来是上层的事**。这样后续支持 HF / vLLM weight / 甚至 TensorRT 都能复用同一套 engine。

另外顺手修了一个小问题：之前我写过 `self.model.dtype`（其实 `nn.Module` 没有这个属性），这里直接用 `next(self.model.parameters()).dtype` 拿 dtype 更稳。

## Benchmark：新增 --hf-model-id，一键跑起来

为了让对比更方便，我给 `benchmark_scheduler.py` 加了一个 `--hf-model-id` 参数：有这个参数就走 HF 加载；否则还是走 checkpoint 路径。

最开始我把 `build_tokenizer(args.tokenizer_name)` 写在了分支外面，结果 `--hf-model-id` 模式下 `tokenizer_name=None`，transformers 就跑去请求 `https://huggingface.co/None/...` 直接 404（非常经典的“提前初始化导致分支参数不完整”）。

修复方式也很直观：**把 tokenizer 初始化和 prompt_lens 计算下沉到分支里**：

```diff
diff --git a/rosellm/roseinfer/benchmark_scheduler.py b/rosellm/roseinfer/benchmark_scheduler.py
index 6429e73..22d08d3 100644
--- a/rosellm/roseinfer/benchmark_scheduler.py
+++ b/rosellm/roseinfer/benchmark_scheduler.py
@@ -18,16 +19,20 @@ def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(
         description="Benchmark the scheduler",
     )
+    parser.add_argument(
+        "--hf-model-id",
+        type=str,
+        help="HF model ID",
+    )
@@ -573,41 +578,83 @@ def benchmark_online(
 def main() -> None:
     args = parse_args()
+    if args.hf_model_id is None and args.checkpoint_path is None:
+        raise SystemExit("either --hf-model-id or --checkpoint-path must be provided")
     prompts = build_prompts(args)
     block_size = 64
-    tokenizer = build_tokenizer(args.tokenizer_name)
-    prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
+    if args.hf_model_id is not None:
+        model, cfg, tokenizer = load_gpt2_from_hf_pretrained(...)
+        prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
+        engine = InferenceEngine(model=model, config=cfg, tokenizer=tokenizer, ...)
+    else:
+        if args.tokenizer_name is None:
+            raise SystemExit("--tokenizer-name is required when using --checkpoint-path")
+        tokenizer = build_tokenizer(args.tokenizer_name)
+        prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
+        engine = InferenceEngine(checkpoint_path=args.checkpoint_path, ...)
```

同时我把 `rosellm/benchmark_scheduler.sh` 改成默认跑 HF gpt2（这样后续我想换模型时也只用改一行 `--hf-model-id xxx`）：

```diff
diff --git a/rosellm/benchmark_scheduler.sh b/rosellm/benchmark_scheduler.sh
index a80f5bf..2694ac1 100755
--- a/rosellm/benchmark_scheduler.sh
+++ b/rosellm/benchmark_scheduler.sh
@@ -2,8 +2,7 @@
 echo "benchmarking scheduler with prefix cache"
 set -euo pipefail
 COMMON_ARGS=(
---checkpoint-path rosellm/rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt \
---tokenizer-name gpt2 \
+--hf-model-id gpt2 \
 --device cuda \
 ...
 )
```

## 单测：验证 logits 对齐

最后我加了一个完全离线的单测：用 `transformers.GPT2Config` 随机初始化一个 HF GPT2，再把它 state_dict 映射到我们自己的 `GPTModel`，对同一输入比较 logits 最大误差。

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from rosellm.rosetrainer.hf_gpt2 import convert_hf_gpt2_state_dict, gpt_config_from_hf_gpt2
from rosellm.rosetrainer.model import GPTModel

def test_convert_hf_gpt2_state_dict_logits_match() -> None:
    torch.manual_seed(0)
    hf_cfg = GPT2Config(
        vocab_size=128, n_positions=64, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        activation_function="gelu_new", use_cache=False,
    )
    hf = GPT2LMHeadModel(hf_cfg).eval()

    cfg = gpt_config_from_hf_gpt2(hf_cfg)
    ours = GPTModel(cfg).eval()
    ours.load_state_dict(convert_hf_gpt2_state_dict(hf.state_dict(), n_layers=cfg.n_layers))

    input_ids = torch.randint(0, hf_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        hf_logits = hf(input_ids).logits
        our_logits, _ = ours(input_ids, attention_mask=None, labels=None, use_cache=False)
    assert (hf_logits - our_logits).abs().max().item() < 1e-4
```

这类测试的好处是：不用联网、不依赖真实权重下载，CI 也能稳定跑，非常适合做 mapping correctness 的回归。

## 执行结果

现在我们跑 benchmark 就可以直接用 HF gpt2：

```shell
(/data/projects/rosellm/.conda) wine@wine-MS-7D90:/data/projects/rosellm$ ./rosellm/benchmark_scheduler.sh 
benchmarking scheduler with prefix cache
benchmarking naive scheduler
=== naive ===
Requests: 64
Elapsed: 3.483946 seconds
Prompt tokens: 54784
Completion tokens: 836
Total tokens: 55620
Throughput (completion,total): 239.96 tokens/s
Throughput (total,total): 15964.65 tokens/s

benchmarking offline scheduler
=== offline summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 2.765803/2.626733 s
Total time p50/mean: 2.852916/2.715167 s
Throughput(completion,decode) p50/mean: 322.02/321.01 tokens/s
Throughput(completion,total) p50/mean: 311.94/310.50 tokens/s

benchmarking offline scheduler with paged attention
=== offline summary ===
Warmup runs: 1
Measured runs: 3
Decode time p50/mean: 0.497647/0.493404 s
Total time p50/mean: 0.585827/0.580352 s
Throughput(completion,decode) p50/mean: 1700.00/1689.56 tokens/s
Throughput(completion,total) p50/mean: 1444.11/1436.29 tokens/s

benchmarking online scheduler
=== online ===
Requests: 64
Elapsed (prefill/add): 0.294421 seconds
Elapsed (decode/run): 3.902135 seconds
Elapsed (total): 4.196555 seconds
Prompt tokens: 54784
Completion tokens: 861
Total tokens: 55645
Throughput (completion,total): 205.17 tokens/s
Throughput (completion,decode): 220.65 tokens/s
Throughput (total,total): 13259.68 tokens/s

[profile] wrote: profiles/online_decode.json
benchmarking online scheduler with paged attention
=== online ===
Requests: 64
Elapsed (prefill/add): 0.283281 seconds
Elapsed (decode/run): 1.113222 seconds
Elapsed (total): 1.396503 seconds
Prompt tokens: 54784
Completion tokens: 764
Total tokens: 55548
Throughput (completion,total): 547.08 tokens/s
Throughput (completion,decode): 686.30 tokens/s
Throughput (total,total): 39776.49 tokens/s

[profile] wrote: profiles_paged_attn/online_decode.json
```

到这里，我们就有了一个最小闭环：**HF 模型加载 → 复用我们的调度/缓存 → 用同一套 benchmark 方式跑性能**。

后续如果要上 Qwen3，就可以先专注在“模型结构的差异”（RoPE/RMSNorm/GQA…）本身，而不是被一堆工程杂事打断节奏。 

