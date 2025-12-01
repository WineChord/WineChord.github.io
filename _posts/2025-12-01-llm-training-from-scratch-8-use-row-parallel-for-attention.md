---
classes: wide2
title: "从零实现 LLM Training：8. Use Row Parallel for Attention"
excerpt: "将 Row Parallel Linear 引入 Attention 层，搭配 Column Parallel 完成张量并行。"
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

# 从零实现 LLM Training：8. Use Row Parallel for Attention

之前我们实现了 Row Parallel Linear，并将其应用到了 FFN 上，本文对应的 PR 将把他用到 Attention Layer 上，这个 PR 依然会比较简单，是直接把 Row Parallel Linear 来替换 out_proj，并且保持，QKV 的 Column Parallel Linear 的 gather_output 为 true，Row Parallel Linear 的 input_is_parallel 为 false，这意味着 Attention layer 会有一次 all-gather 加一次 all-reduce，理想情况下 QKV 应该是按照 head 维度进行切分的，我们在下一个 PR 会做这种切分，到时候 gather_output 就会是 false，input_is_parallel 就会是 true 了。

## `model.py`

```python
diff --git a/rosellm/rosetrainer/model.py b/rosellm/rosetrainer/model.py
index d95add6..102e308 100644
--- a/rosellm/rosetrainer/model.py
+++ b/rosellm/rosetrainer/model.py
@@ -28,9 +28,15 @@ class MultiHeadSelfAttention(nn.Module):
                 bias=True,
                 gather_output=True,
             )
+            self.out_proj = RowParallelLinear(
+                in_features=config.d_model,
+                out_features=config.d_model,
+                bias=True,
+                input_is_parallel=False,
+            )
         else:
             self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
-        self.out_proj = nn.Linear(config.d_model, config.d_model)
+            self.out_proj = nn.Linear(config.d_model, config.d_model)
         self.dropout = nn.Dropout(config.dropout)
         self.register_buffer(
             "mask",
```

## `test_attention_tp_vs_dense.py`

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from config import GPTConfig
from model import MultiHeadSelfAttention
from tensor_parallel import init_tensor_parallel


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def build_attention(config_base: GPTConfig, device: torch.device):
    dense_cfg = GPTConfig(
        vocab_size=config_base.vocab_size,
        max_position_embeddings=config_base.max_position_embeddings,
        n_layers=config_base.n_layers,
        n_heads=config_base.n_heads,
        d_model=config_base.d_model,
        d_ff=config_base.d_ff,
        dropout=config_base.dropout,
        use_tensor_parallel=False,
    )
    attn_dense = MultiHeadSelfAttention(dense_cfg).to(device)
    tp_cfg = GPTConfig(
        vocab_size=config_base.vocab_size,
        max_position_embeddings=config_base.max_position_embeddings,
        n_layers=config_base.n_layers,
        n_heads=config_base.n_heads,
        d_model=config_base.d_model,
        d_ff=config_base.d_ff,
        dropout=config_base.dropout,
        use_tensor_parallel=True,
    )
    attn_tp = MultiHeadSelfAttention(tp_cfg).to(device)
    return attn_dense, attn_tp


def copy_qkv_from_dense_to_tp(
    attn_dense: MultiHeadSelfAttention,
    attn_tp: MultiHeadSelfAttention,
    world_size: int,
    rank: int,
):
    with torch.no_grad():
        linear_dense: nn.Linear = attn_dense.qkv_proj
        col_tp = attn_tp.qkv_proj
        out_features = linear_dense.out_features
        out_per_rank = out_features // world_size
        start = rank * out_per_rank
        end = start + out_per_rank
        col_tp.weight.copy_(linear_dense.weight[start:end, :])
        if col_tp.bias is not None:
            col_tp.bias.copy_(linear_dense.bias[start:end])


def copy_out_proj_from_dense_to_tp(
    attn_dense: MultiHeadSelfAttention,
    attn_tp: MultiHeadSelfAttention,
    world_size: int,
    rank: int,
):
    with torch.no_grad():
        linear_dense: nn.Linear = attn_dense.out_proj
        row_tp = attn_tp.out_proj
        in_features = linear_dense.in_features
        in_per_rank = in_features // world_size
        start = rank * in_per_rank
        end = start + in_per_rank
        row_tp.weight.copy_(linear_dense.weight[:, start:end])
        if row_tp.bias is not None:
            row_tp.bias.copy_(linear_dense.bias)


def main():
    device, local_rank = setup_distributed()
    init_tensor_parallel()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        print(f"world_size = {world_size}")
    base_cfg = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=1,
        n_heads=4,
        d_model=64,
        d_ff=256,
        dropout=0.0,
    )
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    attn_dense, attn_tp = build_attention(base_cfg, device)
    copy_qkv_from_dense_to_tp(attn_dense, attn_tp, world_size, rank)
    copy_out_proj_from_dense_to_tp(attn_dense, attn_tp, world_size, rank)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, base_cfg.d_model, device=device)
    attention_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=torch.long,
        device=device,
    )
    attn_dense.eval()
    attn_tp.eval()
    with torch.no_grad():
        y_dense = attn_dense(x, attention_mask=attention_mask)
        y_tp = attn_tp(x, attention_mask=attention_mask)
    diff = (y_dense - y_tp).abs().max()
    diff_val = diff.item()
    if rank == 0:
        print("y_dense shape:", y_dense.shape)
        print("y_tp shape:", y_tp.shape)
        print("max |y_dense - y_tp| = ", diff_val)
    cleanup_distributed()


if __name__ == "__main__":
    main()

```

运行结果：

```shell
$ torchrun --nproc-per-node=2 test_attention_tp_vs_dense.py 
W1127 12:56:30.316000 22986 site-packages/torch/distributed/run.py:792] 
W1127 12:56:30.316000 22986 site-packages/torch/distributed/run.py:792] *****************************************
W1127 12:56:30.316000 22986 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 12:56:30.316000 22986 site-packages/torch/distributed/run.py:792] *****************************************
world_size = 2
y_dense shape: torch.Size([2, 8, 64])
y_tp shape: torch.Size([2, 8, 64])
max |y_dense - y_tp| =  2.384185791015625e-07
```
