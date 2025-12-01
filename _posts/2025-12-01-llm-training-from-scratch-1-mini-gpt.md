---
classes: wide2
title: "从零实现 LLM Training：1. mini-GPT"
excerpt: "从零实现 mini-GPT 的配置、模型和简单前向。"
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

这个系列旨在学以致用，目标是从零一步步地写出一个 LLM 分布式训练框架，通过这个过程来学习 Megatron-LM / DeepSpeed 等业界成熟框架的原理。在实现过程中会尽可能保持简洁，每个 PR 保持最小可理解的修改量，避免一头扎进代码的海洋中陷入迷途。

对于第一个 PR 而言，主要包含以下三个文件：

* `config.py` 包含了一些 LLM 训练所需要的最基础的配置项，比如词表的大小，最长的序列长度，模型的层数，Attention Head 的数量，模型的 dimension，FFN 的 dimension，dropout 等
* `model.py` 包含了一个最基本的 GPT 模型的实现，首先实现最重要的 `MultiHeadSelfAttention`，然后是 `FeedForward`，用这两者拼出来 `TransformerBlock`，最后再拼出来 `GPTModel`
* `test_forward.py` 用于简单的测试，确保代码执行符合预期，第一个 PR 先只看 forward

下面依次展开具体代码。

## `config.py`

```python
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

```

表示模型的一些基础配置。

## `model.py`

```python
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.max_position_embeddings, config.max_position_embeddings
                )
            )
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = q @ k.transpose(-2, -1) * self.d_head**-0.5
        causal_mask = self.mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:  # padding mask
            attn_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.d_model)
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        token_emb = self.token_embedding(input_ids)  # [B, T, D]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
        pos_emb = self.position_embedding(position_ids)  # [1, T, D]
        pos_emb = pos_emb.expand(bsz, seq_len, -1)  # [B, T, D]
        x = token_emb + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

```

GPT 模型的经典实现，使用了 Pre-LN，其中 Attention 部分是最复杂的，其次是 GPTModel 部分处理 token/position embedding 的地方，其他地方都还好。这里的 position embedding 采用可学习的 embedding，后期改成 sinusoidal 或者 RoPE 的时候，也会比较复杂。

## `test_forward.py`

```python
import torch
from config import GPTConfig
from model import GPTModel


def main():
    config = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
    )
    model = GPTModel(config)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    logits = model(input_ids, attention_mask=attention_mask)
    print("input_ids shape:", input_ids.shape)  # [B, T]
    print("logits shape:", logits.shape)  # [B, T, V]


if __name__ == "__main__":
    main()

```

这里就是一个简单的 forward 流程，确保可以运行。

## 运行

执行结果类似下面：

```python
$ python test_forward.py 
input_ids shape: torch.Size([2, 16])
logits shape: torch.Size([2, 16, 10000])
```
