---
classes: wide2
title: "从零实现 LLM Training：5. Use Column Parallel"
excerpt: "把 Column Parallel Linear 集成到 GPTModel 中并用训练循环验证。"
categories: 
  - LLM
  - Training
tags: 
  - LLM
  - Training
toc: true
toc_sticky: true
mathjax: true
---

在我们上一个 PR 中已经实现了 Column Parallel Linear，这个 PR 里我们把它集成到 GPTModel，并用 train loop 去验证他。

## `config.py`

```diff
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 32000
    max_position_embeddings: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
+   use_tensor_parallel: bool = False
```

只需要添加一个 `use_tensor_parallel` 的 flag 即可。

## `model.py`

```diff
diff --git a/rosellm/rosetrainer/model.py b/rosellm/rosetrainer/model.py
index 9025c71..fcc20d8 100644
--- a/rosellm/rosetrainer/model.py
+++ b/rosellm/rosetrainer/model.py
@@ -1,9 +1,11 @@
 from typing import Optional
 
 import torch
+import torch.distributed as dist
 import torch.nn as nn
 import torch.nn.functional as F
 from config import GPTConfig
+from tensor_parallel import ColumnParallelLinear, init_tensor_parallel
 
 
 class MultiHeadSelfAttention(nn.Module):
@@ -13,7 +15,17 @@ class MultiHeadSelfAttention(nn.Module):
         self.d_model = config.d_model
         self.n_heads = config.n_heads
         self.d_head = config.d_model // config.n_heads
-        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
+        use_tp = getattr(config, "use_tensor_parallel", False)
+        if use_tp and dist.is_available() and dist.is_initialized():
+            init_tensor_parallel()
+            self.qkv_proj = ColumnParallelLinear(
+                in_features=config.d_model,
+                out_features=3 * config.d_model,
+                bias=True,
+                gather_output=True,
+            )
+        else:
+            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
         self.out_proj = nn.Linear(config.d_model, config.d_model)
         self.dropout = nn.Dropout(config.dropout)
         self.register_buffer(
@@ -57,7 +69,17 @@ class MultiHeadSelfAttention(nn.Module):
 class FeedForward(nn.Module):
     def __init__(self, config: GPTConfig):
         super().__init__()
-        self.fc1 = nn.Linear(config.d_model, config.d_ff)
+        use_tp = getattr(config, "use_tensor_parallel", False)
+        if use_tp and dist.is_available() and dist.is_initialized():
+            init_tensor_parallel()
+            self.fc1 = ColumnParallelLinear(
+                in_features=config.d_model,
+                out_features=config.d_ff,
+                bias=True,
+                gather_output=True,
+            )
+        else:
+            self.fc1 = nn.Linear(config.d_model, config.d_ff)
         self.fc2 = nn.Linear(config.d_ff, config.d_model)
         self.dropout = nn.Dropout(config.dropout)
 
```

在 Attention layer 的 QKV 参数矩阵以及 FFN layer 的第一个参数矩阵加一个判断来使用 Column Parallel Linear。

此处由于仅有 Column Parallel Linear，因此每一层都在频繁地 all-gather（但是其实即使还有了 Row Parallel Linear，每一层仍然会频繁地 all-reduce）。

## `test_forward_tp.py`

```python
import os

import torch
import torch.distributed as dist
from config import GPTConfig
from model import GPTModel
from tensor_parallel import init_tensor_parallel


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def main():
    device, local_rank = setup_distributed()
    torch.manual_seed(123)  # for deterministic seed
    torch.cuda.manual_seed(123)  # for deterministic seed
    init_tensor_parallel()
    world_size = dist.get_world_size()
    if local_rank == 0:
        print("world_size:", world_size)
    config = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
        use_tensor_parallel=True,
    )
    model = GPTModel(config).to(device)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=torch.long,
        device=device,
    )
    logits, loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    with torch.no_grad():
        logits_ref = logits.clone()
        dist.broadcast(logits_ref, src=0)
        diff = (logits_ref - logits).abs().max()
        diff_val = diff.item()
    print("max diff vs rank0:", diff_val)
    if local_rank == 0:
        print("input_ids shape:", input_ids.shape)
        print("logits shape:", logits.shape)
        print("loss:", loss.item())
    cleanup_distributed()


if __name__ == "__main__":
    main()

```

此处要注意使用 `torch.manual_seed` 以及`torch.cuda.manual_seed` 来保证每张卡用的 `input_ids` 是随机生成的一样的，并且 dropout 的确定性。

## 运行

```shell
$ torchrun --nproc-per-node=2 test_forward_tp.py 
W1126 20:20:40.188000 3266158 site-packages/torch/distributed/run.py:792] 
W1126 20:20:40.188000 3266158 site-packages/torch/distributed/run.py:792] *****************************************
W1126 20:20:40.188000 3266158 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 20:20:40.188000 3266158 site-packages/torch/distributed/run.py:792] *****************************************
world_size: 2
max diff vs rank0: 0.0
input_ids shape: torch.Size([2, 16])
logits shape: torch.Size([2, 16, 10000])
loss: 9.380536079406738
max diff vs rank0: 0.0
```
