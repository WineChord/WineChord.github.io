---
classes: wide2
title: "从零实现 LLM Training：2. Loss and Train Loop"
excerpt: "为 mini-GPT 加上最基本的 loss 和最小 train loop。"
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

# 从零实现 LLM Training：2. Loss and Train Loop

在实现了第一步 mini-GPT 后，第二个 PR 我们来添加下最基本的 loss，并写一个最小的 train loop，使用一个非常简单的 toy 数据集。

我们已经有了 `config.py, model.py, test_forward.py` 三个文件，依次实现了配置、模型、简单前向。

为了写一个最小的 train loop，我们需要模型能够返回一个 loss，这样我们才能在 loss 上做 backward。

## `model.py`

首先我们修改 `model.py`

```diff
diff --git a/model.py b/model.py
index e1b7abc..9d2f4cd 100644
--- a/model.py
+++ b/model.py
@@ -102,19 +102,33 @@ class GPTModel(nn.Module):
     def forward(
         self,
         input_ids: torch.Tensor,  # [B, T]
-        attention_mask: Optional[torch.Tensor] = None,
+        attention_mask: Optional[torch.Tensor] = None,
+        labels: Optional[torch.Tensor] = None,
     ):
         bsz, seq_len = input_ids.size()
         device = input_ids.device
         token_emb = self.token_embedding(input_ids)              # [B, T, D]
         position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
         pos_emb = self.position_embedding(position_ids)                   # [1, T, D]
         pos_emb = pos_emb.expand(bsz, seq_len, -1)                # [B, T, D]
         x = token_emb + pos_emb
         x = self.dropout(x)

         for block in self.blocks:
             x = block(x, attention_mask=attention_mask)

         x = self.ln_f(x)
         logits = self.lm_head(x)

-        return logits
+        loss = None
+        if labels is not None:
+            # shift so that tokens predict the next token
+            shift_logits = logits[:, :-1, :].contiguous()        # [B, T-1, V]
+            shift_labels = labels[:, 1:].contiguous()            # [B, T-1]
+
+            loss = F.cross_entropy(
+                shift_logits.view(-1, self.config.vocab_size),   # [B*(T-1), V]
+                shift_labels.view(-1),                           # [B*(T-1)]
+            )
+
+        return logits, loss      # [B, T, V], scalar loss

```

为了产生 loss，我们需要传入 `labels`，然后 logits 与 labels 需要做移位处理以构造 next token prediction 的效果。

## `test_forward.py`

由于我们入参添加了 labels，出参多了 loss，所以前向的测试也需要做一些修改：

```diff
diff --git a/test_forward.py b/test_forward.py
index 2a1bc37..73fe0ae 100644
--- a/test_forward.py
+++ b/test_forward.py
@@ -23,9 +23,17 @@ def main():
     attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

-    logits = model(input_ids, attention_mask=attention_mask)
+    logits, loss = model(
+        input_ids,
+        attention_mask=attention_mask,
+        labels=input_ids,  # For tmp test
+    )

     print("input_ids shape:", input_ids.shape)   # [B, T]
     print("logits shape:", logits.shape)         # [B, T, V]
+    print("loss:", loss.item())

 if __name__ == "__main__":
     main()

```

## `train_minimal.py`

然后我们可以做一个非常简单的 train loop：

```python 
import torch
from config import GPTConfig
from model import GPTModel
from torch.utils.data import DataLoader, Dataset


class ToyRandomDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            dtype=torch.long,
        )
        labels = input_ids.clone()
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = GPTConfig(
        vocab_size=10000,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
    )
    model = GPTModel(config).to(device)
    dataset = ToyRandomDataset(
        vocab_size=config.vocab_size,
        seq_len=32,
        num_samples=1000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    num_steps = 50
    step = 0
    for batch in dataloader:
        step += 1
        if step > num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step {step} / {num_steps} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
```

为了简单起见，数据集是一个随机生成的数据集，运行这个文件来初步判断流程正确。

## 运行

运行结果如下：

```shell
$ python train_minimal.py 
Using device: cuda
step 10 / 50 | loss: 9.3910
step 20 / 50 | loss: 9.4008
step 30 / 50 | loss: 9.3672
step 40 / 50 | loss: 9.3834
step 50 / 50 | loss: 9.4179
```

由于是随机数据集，所以 loss 不会下降，之后我们使用真实数据集的话，可以观察到 loss 是会下降的。

后续分布式训练的各种新能力也会以这个 PR 为基础进行迭代，从而我们可以观察到一个项目是如何从零到一的全过程。
