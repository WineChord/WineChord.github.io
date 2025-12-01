---
classes: wide2
title: "从零实现 LLM Training：16. Multi Doc Dataset"
excerpt: "重构 Dataset，支持多文档输入与更合理的切分策略。"
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

我们已经实现了张量并行、混合精度、checkpoint、eval and logging，并使用了真实数据做了一些小训练，并且做了很简单的生成，这个 PR 可以来好好修一修目前项目中 Dataset 不合理的处理，比如目前是每行后加 eos，是不合理的，我们自然引入多文档（多 txt），可以在每个文档本身看成一个整体，只在文档之间加 eos，并且我们之前取 index 的位置实际上是固定的（需要是 seq_len 的倍数），这里也是不合理的，应该以任意位置都可以做起点。

本文对应 PR 的整体目标还是简洁，能够满足最小的需求，大概是处理上百篇 txt 这种级别，再高的话需要引入 megatron 的工业级方案：

* 分 IndexedDataset 存 .bin, .idx（前者是所有 doc 平铺的 tokenize 之后的结果，后者则用于标识每个 doc 对应在 .bin 中的起始位置以及长度）
* 分 GPTDataset 存：
  * Do_idx：多个文档 id 重复 shuffle，用来模拟随机的文档流
  * Sa_idx：sample index，每一项是 doc_index_pos, offset_in_that_doc，表示这个 sample 是从 Do_idx 的哪个下标开始的，并且位于该文档中的哪个 offset 上，相邻两项 Sa_idx 就可以确定这个 sample 对应的区间，注意这里明显可以看到，sample 可以跨文档
  * Sh_idx：shuffle index，相当于实际取第 k 个样本的时候取的是 Sa_idx[Sh_idx[k]]，使样本顺序随机

* 分 BlendedDataset 去把不同 corpus 做一些融合策略

简单起见，本 PR 还是做内存版，即把所有 tokenize 好的 token 存到内存的一个 all_ids 中，然后对 start 位置做随机 sample

## `dataset.py`

```diff
diff --git a/rosellm/rosetrainer/dataset.py b/rosellm/rosetrainer/dataset.py
index 9a9fc99..b0970d2 100644
--- a/rosellm/rosetrainer/dataset.py
+++ b/rosellm/rosetrainer/dataset.py
@@ -1,4 +1,5 @@
-from typing import List
+import random
+from typing import List, Optional
 
 import torch
 from torch.utils.data import Dataset
@@ -31,46 +32,64 @@ class TextDatasetForCausalLM(Dataset):
         tokenizer,
         seq_len: int,
         add_eos: bool = True,
+        max_tokens: Optional[int] = None,
+        seed: Optional[int] = None,
     ) -> None:
         super().__init__()
         self.tokenizer = tokenizer
         self.seq_len = seq_len
-        self.add_eos = add_eos
-        texts: List[str] = []
+        eos_id: Optional[int] = None
+        if add_eos and hasattr(tokenizer, "eos_token_id"):
+            eos_id = tokenizer.eos_token_id
+        all_ids: List[int] = []
+        total_tokens = 0
+        total_files = 0
         for path in file_paths:
+            total_files += 1
             with open(path, "r", encoding="utf-8") as f:
-                for line in f:
-                    line = line.strip()
-                    if not line:
-                        continue
-                    texts.append(line)
-        if not texts:
-            raise ValueError("the file is full of empty lines")
-        all_ids: List[int] = []
-        for text in texts:
-            if add_eos and hasattr(tokenizer, "eos_token_id"):
-                encoded = tokenizer.encode(
-                    text,
-                    add_special_tokens=False,
-                )
-                all_ids.extend(encoded)
-                all_ids.append(tokenizer.eos_token_id)
-            else:
-                encoded = tokenizer.encode(
-                    text,
-                    add_special_tokens=False,
-                )
-                all_ids.extend(encoded)
+                text = f.read()
+            if not text.strip():
+                continue
+            ids = tokenizer.encode(text, add_special_tokens=False)
+            if not ids:
+                continue
+            all_ids.extend(ids)
+            total_tokens += len(ids)
+            if eos_id is not None:
+                all_ids.append(eos_id)
+                total_tokens += 1
+            if max_tokens is not None and total_tokens >= max_tokens:
+                all_ids = all_ids[:max_tokens]
+                total_tokens = len(all_ids)
+                break
+        print(f"total files: {total_files}")
+        print(f"total tokens: {total_tokens}")
         if len(all_ids) < seq_len:
             raise ValueError("the total number of tokens is less than seq_len")
         self.all_ids = all_ids
-        self.num_samples = len(all_ids) // seq_len
+        self.total_tokens = len(all_ids)
+        max_start = self.total_tokens - seq_len
+        num_samples = self.total_tokens // seq_len
+        if num_samples > max_start + 1:
+            num_samples = max_start + 1
+        if seed is None:
+            rng = random.Random()
+        else:
+            rng = random.Random(seed)
+        candidates = list(range(max_start + 1))
+        if num_samples < len(candidates):
+            start_indices = rng.sample(candidates, num_samples)
+        else:
+            rng.shuffle(candidates)
+            start_indices = candidates
+        self.start_indices = start_indices
+        self.num_samples = len(self.start_indices)
 
     def __len__(self) -> int:
         return self.num_samples
 
     def __getitem__(self, idx: int):
-        start = idx * self.seq_len
+        start = self.start_indices[idx]
         end = start + self.seq_len
         ids = self.all_ids[start:end]
         input_ids = torch.tensor(ids, dtype=torch.long)

```



## `train_minimal.py`

```python
diff --git a/rosellm/rosetrainer/train_minimal.py b/rosellm/rosetrainer/train_minimal.py
index ed74147..8553d00 100644
--- a/rosellm/rosetrainer/train_minimal.py
+++ b/rosellm/rosetrainer/train_minimal.py
@@ -130,12 +130,18 @@ def main(args: argparse.Namespace) -> None:
             file_paths=args.train_data,
             tokenizer=tokenizer,
             seq_len=args.seq_len,
+            add_eos=True,
+            max_tokens=args.max_tokens,
+            seed=args.data_seed,
         )
         if args.val_data:
             val_dataset = TextDatasetForCausalLM(
                 file_paths=args.val_data,
                 tokenizer=tokenizer,
                 seq_len=args.seq_len,
+                add_eos=True,
+                max_tokens=args.max_tokens,
+                seed=args.data_seed + 1,
             )
         else:
             val_size = max(int(0.1 * len(train_dataset)), 1)
@@ -342,6 +348,18 @@ def parse_args() -> argparse.Namespace:
         action="store_true",
         help="use random toy data rather than real data",
     )
+    parser.add_argument(
+        "--max-tokens",
+        type=int,
+        default=None,
+        help="max tokens to sample from the dataset",
+    )
+    parser.add_argument(
+        "--data-seed",
+        type=int,
+        default=None,
+        help="seed for the data sampler",
+    )
     return parser.parse_args()
```

## `train_ddp.py`

```python
diff --git a/rosellm/rosetrainer/train_ddp.py b/rosellm/rosetrainer/train_ddp.py
index fcfe19e..867ca31 100644
--- a/rosellm/rosetrainer/train_ddp.py
+++ b/rosellm/rosetrainer/train_ddp.py
@@ -163,12 +163,18 @@ def main(args: argparse.Namespace) -> None:
             file_paths=args.train_data,
             tokenizer=tokenizer,
             seq_len=args.seq_len,
+            add_eos=True,
+            max_tokens=args.max_tokens,
+            seed=args.data_seed,
         )
         if args.val_data:
             val_dataset = TextDatasetForCausalLM(
                 file_paths=args.val_data,
                 tokenizer=tokenizer,
                 seq_len=args.seq_len,
+                add_eos=True,
+                max_tokens=args.max_tokens,
+                seed=args.data_seed + 1,
             )
         else:
             val_size = max(int(0.1 * len(train_dataset)), 1)
@@ -411,6 +417,18 @@ def parse_args() -> argparse.Namespace:
         action="store_true",
         help="use toy data",
     )
+    parser.add_argument(
+        "--max-tokens",
+        type=int,
+        default=None,
+        help="max tokens to sample from the dataset",
+    )
+    parser.add_argument(
+        "--data-seed",
+        type=int,
+        default=None,
+        help="seed for the data sampler",
+    )
     return parser.parse_args()
```

## 运行

可以搞多个 gutenberg 的 txt 进行训练：

```shell
#!/bin/bash
torchrun --nproc_per_node=2 train_ddp.py \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.1 \
  --max-position-embeddings 1024 \
  --seq-len 1024 \
  --batch-size 2 \
  --num-steps 6000 \
  --lr 3e-4 \
  --train-data data/gutenberg/*.txt \
  --tokenizer-name gpt2 \
  --max-tokens 10000000 \
  --data-seed 42 \
  --checkpoint-path checkpoints/gpt2_small_ddp.pt
```

形如：

```shell
('epoch 1 step 760 / 6000 ', 'train loss: 5.8938 ', 'val loss: 5.5948 ', 'val ppl: 269.0184 ', 'amp: True')
('epoch 1 step 770 / 6000 ', 'train loss: 6.0291 ', 'val loss: 5.5967 ', 'val ppl: 269.5446 ', 'amp: True')
('epoch 1 step 780 / 6000 ', 'train loss: 5.6675 ', 'val loss: 5.5812 ', 'val ppl: 265.3791 ', 'amp: True')
('epoch 1 step 790 / 6000 ', 'train loss: 5.9663 ', 'val loss: 5.5908 ', 'val ppl: 267.9512 ', 'amp: True')
('epoch 1 step 800 / 6000 ', 'train loss: 5.0629 ', 'val loss: 5.5759 ', 'val ppl: 263.9961 ', 'amp: True')
('epoch 1 step 810 / 6000 ', 'train loss: 6.4135 ', 'val loss: 5.5756 ', 'val ppl: 263.9148 ', 'amp: True')
('epoch 1 step 820 / 6000 ', 'train loss: 5.8329 ', 'val loss: 5.5482 ', 'val ppl: 256.7634 ', 'amp: True')
('epoch 1 step 830 / 6000 ', 'train loss: 4.9374 ', 'val loss: 5.5305 ', 'val ppl: 252.2779 ', 'amp: True')
('epoch 1 step 840 / 6000 ', 'train loss: 3.3967 ', 'val loss: 5.5210 ', 'val ppl: 249.8808 ', 'amp: True')
('epoch 1 step 850 / 6000 ', 'train loss: 5.2579 ', 'val loss: 5.5221 ', 'val ppl: 250.1612 ', 'amp: True')
('epoch 1 step 860 / 6000 ', 'train loss: 5.2088 ', 'val loss: 5.5418 ', 'val ppl: 255.1386 ', 'amp: True')
```
