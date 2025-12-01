---
classes: wide2
title: "从零实现 LLM Training：14. Use Real Data"
excerpt: "将 toy 数据集替换为更真实的文本数据，完善训练链路。"
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

前面我们已经实现了张量并行，混合精度训练，checkpoint，argparse，eval and logging，现在我们可以考虑把数据搞成真实一些的，而不是只在 toy 上玩。

## `dataset.py`

```python
from typing import List

import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def build_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
):
    if AutoTokenizer is None:
        raise ImportError("need to install transformers")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TextDatasetForCausalLM(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        seq_len: int,
        add_eos: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_eos = add_eos
        texts: List[str] = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(line)
        if not texts:
            raise ValueError("the file is full of empty lines")
        all_ids: List[int] = []
        for text in texts:
            if add_eos and hasattr(tokenizer, "eos_token_id"):
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
                all_ids.extend(encoded)
                all_ids.append(tokenizer.eos_token_id)
            else:
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
                all_ids.extend(encoded)
        if len(all_ids) < seq_len:
            raise ValueError("the total number of tokens is less than seq_len")
        self.all_ids = all_ids
        self.num_samples = len(all_ids) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        ids = self.all_ids[start:end]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

```

核心实际上就是单独的 Dataset class，这里为了简单起见，直接用的 huggingface 的 tokenizer，并且切的方法也比较粗暴，就是按照 seq_len 来切成一段一段，一般业界的训练而言应该是可以以任意位置为起点去取一个 seq_len 的，可以作为后续优化，第一个数据相关的 PR 先尽可能地简单。

## `train_simple.py`

```diff
diff --git a/rosellm/rosetrainer/train_minimal.py b/rosellm/rosetrainer/train_minimal.py
index a8883cd..ed74147 100644
--- a/rosellm/rosetrainer/train_minimal.py
+++ b/rosellm/rosetrainer/train_minimal.py
@@ -6,6 +6,7 @@ from datetime import datetime
 import torch
 from checkpoint import load_checkpoint, save_checkpoint
 from config import GPTConfig
+from dataset import TextDatasetForCausalLM, build_tokenizer
 from model import GPTModel
 from torch.amp import GradScaler, autocast
 from torch.utils.data import DataLoader, Dataset
@@ -40,6 +41,7 @@ def log_line(path: str, text: str | tuple[str, ...]) -> None:
     os.makedirs(os.path.dirname(path), exist_ok=True)
     with open(path, "a", encoding="utf-8") as f:
         f.write(str(text) + "\n")
+    print(text)
 
 
 def evaluate(
@@ -88,8 +90,20 @@ def main(args: argparse.Namespace) -> None:
     log_line(log_path, f"Arguments: {args}")
     checkpoint_path = args.checkpoint_path
     resume = args.resume
+
+    if args.use_toy_data:
+        effective_vocab_size = args.vocab_size
+    else:
+        if not args.train_data:
+            raise ValueError("--train-data is not provided")
+        tokenizer = build_tokenizer(args.tokenizer_name)
+        tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
+        if tokenizer_vocab_size is None:
+            tokenizer_vocab_size = len(tokenizer)
+        effective_vocab_size = tokenizer_vocab_size
+
     config = GPTConfig(
-        vocab_size=args.vocab_size,
+        vocab_size=effective_vocab_size,
         max_position_embeddings=args.max_position_embeddings,
         n_layers=args.n_layers,
         n_heads=args.n_heads,
@@ -98,17 +112,40 @@ def main(args: argparse.Namespace) -> None:
         dropout=args.dropout,
     )
     model = GPTModel(config).to(device)
-    full_dataset = ToyRandomDataset(
-        vocab_size=config.vocab_size,
-        seq_len=args.seq_len,
-        num_samples=1000,
-    )
-    val_size = max(int(0.2 * len(full_dataset)), 1)
-    train_size = len(full_dataset) - val_size
-    train_dataset, val_dataset = torch.utils.data.random_split(
-        full_dataset,
-        [train_size, val_size],
-    )
+
+    if args.use_toy_data:
+        full_dataset = ToyRandomDataset(
+            vocab_size=config.vocab_size,
+            seq_len=args.seq_len,
+            num_samples=1000,
+        )
+        val_size = max(int(0.2 * len(full_dataset)), 1)
+        train_size = len(full_dataset) - val_size
+        train_dataset, val_dataset = torch.utils.data.random_split(
+            full_dataset,
+            [train_size, val_size],
+        )
+    else:
+        train_dataset = TextDatasetForCausalLM(
+            file_paths=args.train_data,
+            tokenizer=tokenizer,
+            seq_len=args.seq_len,
+        )
+        if args.val_data:
+            val_dataset = TextDatasetForCausalLM(
+                file_paths=args.val_data,
+                tokenizer=tokenizer,
+                seq_len=args.seq_len,
+            )
+        else:
+            val_size = max(int(0.1 * len(train_dataset)), 1)
+            train_size = len(train_dataset) - val_size
+            train_dataset, val_dataset = torch.utils.data.random_split(
+                train_dataset,
+                [train_size, val_size],
+            )
+    log_line(log_path, f"train dataset size: {len(train_dataset)}")
+    log_line(log_path, f"val dataset size: {len(val_dataset)}")
     train_dataloader = DataLoader(
         train_dataset,
         batch_size=args.batch_size,
@@ -126,65 +163,68 @@ def main(args: argparse.Namespace) -> None:
     num_steps = args.num_steps
     step = 0
     if resume and os.path.exists(checkpoint_path):
-        print(f"Resuming from checkpoint {checkpoint_path}")
+        log_line(log_path, f"Resuming from checkpoint {checkpoint_path}")
         step, extra = load_checkpoint(checkpoint_path, model, optimizer, scaler)
-        print(f"Resumed from step {step}")
+        log_line(log_path, f"Resumed from step {step}")
     elif resume:
-        print("Resume flag is set, but checkpoint not found. Starting from scratch.")
+        log_line(
+            log_path,
+            "Resume flag is set, but checkpoint not found. Starting from scratch.",
+        )
     else:
-        print("Starting from scratch")
-    for batch in train_dataloader:
-        step += 1
-        if step > num_steps:
-            break
-        input_ids = batch["input_ids"].to(device)
-        labels = batch["labels"].to(device)
-        attention_mask = batch["attention_mask"].to(device)
-        optimizer.zero_grad()
-        if use_amp:
-            with autocast(device_type=device.type):
+        log_line(log_path, "Starting from scratch")
+    while step < num_steps:
+        for batch in train_dataloader:
+            step += 1
+            if step > num_steps:
+                break
+            input_ids = batch["input_ids"].to(device)
+            labels = batch["labels"].to(device)
+            attention_mask = batch["attention_mask"].to(device)
+            optimizer.zero_grad()
+            if use_amp:
+                with autocast(device_type=device.type):
+                    logits, loss = model(
+                        input_ids=input_ids,
+                        attention_mask=attention_mask,
+                        labels=labels,
+                    )
+                scaler.scale(loss).backward()
+                scaler.step(optimizer)
+                scaler.update()
+            else:
                 logits, loss = model(
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     labels=labels,
                 )
-            scaler.scale(loss).backward()
-            scaler.step(optimizer)
-            scaler.update()
-        else:
-            logits, loss = model(
-                input_ids=input_ids,
-                attention_mask=attention_mask,
-                labels=labels,
-            )
-            loss.backward()
-            optimizer.step()
-        if step % 20 == 0:
-            save_checkpoint(
-                checkpoint_path,
-                model=model,
-                optimizer=optimizer,
-                step=step,
-                scaler=scaler if use_amp else None,
-                extra={"note": "single_gpt_minimal"},
-            )
-        if step % 10 == 0:
-            val_loss = evaluate(
-                model,
-                val_dataloader,
-                device=device,
-                use_amp=use_amp,
-            )
-            val_ppl = math.exp(val_loss)
-            msg = (
-                f"step {step} / {num_steps} ",
-                f"train loss: {loss.item():.4f} ",
-                f"val loss: {val_loss:.4f} ",
-                f"val ppl: {val_ppl:.4f} ",
-                f"amp: {use_amp}",
-            )
-            print(msg)
-            log_line(log_path, msg)
+                loss.backward()
+                optimizer.step()
+            if step % 20 == 0:
+                save_checkpoint(
+                    checkpoint_path,
+                    model=model,
+                    optimizer=optimizer,
+                    step=step,
+                    scaler=scaler if use_amp else None,
+                    extra={"note": "single_gpt_minimal"},
+                )
+            if step % 10 == 0:
+                val_loss = evaluate(
+                    model,
+                    val_dataloader,
+                    device=device,
+                    use_amp=use_amp,
+                )
+                val_ppl = math.exp(val_loss)
+                msg = (
+                    f"step {step} / {num_steps} ",
+                    f"train loss: {loss.item():.4f} ",
+                    f"val loss: {val_loss:.4f} ",
+                    f"val ppl: {val_ppl:.4f} ",
+                    f"amp: {use_amp}",
+                )
+                log_line(log_path, msg)
 
 
 def parse_args() -> argparse.Namespace:
@@ -198,7 +238,7 @@ def parse_args() -> argparse.Namespace:
     parser.add_argument(
         "--max-position-embeddings",
         type=int,
-        default=128,
+        default=10000,
         help="Max sequence length.",
     )
     parser.add_argument(
@@ -276,6 +316,32 @@ def parse_args() -> argparse.Namespace:
         action="store_true",
         help="Resume training from checkpoint.",
     )
+    # data and tokenizer
+    parser.add_argument(
+        "--train-data",
+        type=str,
+        nargs="*",
+        default=[],
+        help="Path to training data",
+    )
+    parser.add_argument(
+        "--val-data",
+        type=str,
+        nargs="*",
+        default=[],
+        help="Path to val data (optional, can be auto-split from train)",
+    )
+    parser.add_argument(
+        "--tokenizer-name",
+        type=str,
+        default="gpt2",
+        help="tokenizer name",
+    )
+    parser.add_argument(
+        "--use-toy-data",
+        action="store_true",
+        help="use random toy data rather than real data",
+    )
     return parser.parse_args()
```

单卡训练基本上就是用一下新的这个 Dataset class，然后加一些命令行选项，有了这些，就能做真实的模型训练了。

## `train_ddp.py`

```diff
diff --git a/rosellm/rosetrainer/train_ddp.py b/rosellm/rosetrainer/train_ddp.py
index eb7dfa6..8f3f07d 100644
--- a/rosellm/rosetrainer/train_ddp.py
+++ b/rosellm/rosetrainer/train_ddp.py
@@ -7,6 +7,7 @@ import torch
 import torch.distributed as dist
 from checkpoint import load_checkpoint, save_checkpoint
 from config import GPTConfig
+from dataset import TextDatasetForCausalLM, build_tokenizer
 from model import GPTModel
 from torch.amp import GradScaler, autocast
 from torch.nn.parallel import DistributedDataParallel as DDP
@@ -42,6 +43,7 @@ def log_line(path: str, text: str | tuple[str, ...]) -> None:
     os.makedirs(os.path.dirname(path), exist_ok=True)
     with open(path, "a", encoding="utf-8") as f:
         f.write(str(text) + "\n")
+    print(text)
 
 
 def evaluate_ddp(
@@ -116,8 +118,20 @@ def main(args: argparse.Namespace) -> None:
         log_line(log_path, f"[rank {local_rank}] Using device: {device}")
         log_line(log_path, f"Arguments: {args}")
         os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
+
+    if args.use_toy_data:
+        effective_vocab_size = args.vocab_size
+    else:
+        if not args.train_data:
+            raise ValueError("--train-data is not provided")
+        tokenizer = build_tokenizer(args.tokenizer_name)
+        tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
+        if tokenizer_vocab_size is None:
+            tokenizer_vocab_size = len(tokenizer)
+        effective_vocab_size = tokenizer_vocab_size
+
     config = GPTConfig(
-        vocab_size=args.vocab_size,
+        vocab_size=effective_vocab_size,
         max_position_embeddings=args.max_position_embeddings,
         n_layers=args.n_layers,
         n_heads=args.n_heads,
@@ -132,17 +146,37 @@ def main(args: argparse.Namespace) -> None:
         output_device=device.index,
         find_unused_parameters=False,
     )
-    full_dataset = ToyRandomDataset(
-        vocab_size=config.vocab_size,
-        seq_len=args.seq_len,
-        num_samples=1000,
-    )
-    val_size = max(int(0.2 * len(full_dataset)), 1)
-    train_size = len(full_dataset) - val_size
-    train_dataset, val_dataset = torch.utils.data.random_split(
-        full_dataset,
-        [train_size, val_size],
-    )
+    if args.use_toy_data:
+        full_dataset = ToyRandomDataset(
+            vocab_size=config.vocab_size,
+            seq_len=args.seq_len,
+            num_samples=1000,
+        )
+        val_size = max(int(0.2 * len(full_dataset)), 1)
+        train_size = len(full_dataset) - val_size
+        train_dataset, val_dataset = torch.utils.data.random_split(
+            full_dataset,
+            [train_size, val_size],
+        )
+    else:
+        train_dataset = TextDatasetForCausalLM(
+            file_paths=args.train_data,
+            tokenizer=tokenizer,
+            seq_len=args.seq_len,
+        )
+        if args.val_data:
+            val_dataset = TextDatasetForCausalLM(
+                file_paths=args.val_data,
+                tokenizer=tokenizer,
+                seq_len=args.seq_len,
+            )
+        else:
+            val_size = max(int(0.1 * len(train_dataset)), 1)
+            train_size = len(train_dataset) - val_size
+            train_dataset, val_dataset = torch.utils.data.random_split(
+                train_dataset,
+                [train_size, val_size],
+            )
     train_sampler = DistributedSampler(
         train_dataset,
         num_replicas=dist.get_world_size(),
@@ -173,7 +207,9 @@ def main(args: argparse.Namespace) -> None:
     num_steps = args.num_steps
     step = 0
     if resume and os.path.exists(checkpoint_path):
-        print(f"[rank {local_rank}] Resuming from checkpoint {checkpoint_path}")
+        log_line(
+            log_path, f"[rank {local_rank}] Resuming from checkpoint {checkpoint_path}"
+        )
         step, extra = load_checkpoint(
             checkpoint_path,
             ddp_model.module,
@@ -181,13 +217,14 @@ def main(args: argparse.Namespace) -> None:
             scaler,
             map_location=device.type,
         )
-        print(f"[rank {local_rank}] Resumed from step {step}")
+        log_line(log_path, f"[rank {local_rank}] Resumed from step {step}")
     elif resume and is_main_process(local_rank):
-        print(
-            f"[rank {local_rank}] Resume flag is set, but checkpoint not found. Starting from scratch."
+        log_line(
+            log_path,
+            f"[rank {local_rank}] Resume flag is set, but checkpoint not found. Starting from scratch.",
         )
     elif is_main_process(local_rank):
-        print(f"[rank {local_rank}] Starting from scratch")
+        log_line(log_path, f"[rank {local_rank}] Starting from scratch")
     for epoch in range(1, 1000):
         train_sampler.set_epoch(epoch)
         for batch in train_dataloader:
@@ -241,12 +278,11 @@ def main(args: argparse.Namespace) -> None:
                         f"val ppl: {val_ppl:.4f} ",
                         f"amp: {use_amp}",
                     )
-                    print(msg)
                     log_line(log_path, msg)
         if step > num_steps:
             break
     if is_main_process(local_rank):
-        print("Training finished.")
+        log_line(log_path, "Training finished.")
     cleanup_distributed()
 
 
@@ -261,7 +297,7 @@ def parse_args() -> argparse.Namespace:
     parser.add_argument(
         "--max-position-embeddings",
         type=int,
-        default=128,
+        default=10000,
         help="Max sequence length.",
     )
     parser.add_argument(
@@ -339,6 +375,32 @@ def parse_args() -> argparse.Namespace:
         action="store_true",
         help="Resume training from checkpoint.",
     )
+    # data and tokenizer
+    parser.add_argument(
+        "--train-data",
+        type=str,
+        nargs="*",
+        default=[],
+        help="Path to training data",
+    )
+    parser.add_argument(
+        "--val-data",
+        type=str,
+        nargs="*",
+        default=[],
+        help="Path to val data",
+    )
+    parser.add_argument(
+        "--tokenizer-name",
+        type=str,
+        default="gpt2",
+        help="tokenizer name",
+    )
+    parser.add_argument(
+        "--use-toy-data",
+        action="store_true",
+        help="use toy data",
+    )
     return parser.parse_args()
```

DDP 代码改动也比较类似。

## 运行

这里我们可以搞一些真实的文本语料来进行训练，比如 https://www.gutenberg.org/ebooks/11 提取一个 txt，然后把一大部分放到 data/train.txt，剩下的部分放到 data/val.txt，然后就可以进行训练，可以看到 loss 在下降。这样训练好的模型其实可以在上面做推理看看他输出的效果，我们可以在下一个 PR 加一个简单的推理能力，用于感性的效果查看。

单卡训练：

```shell
$ python train_minimal.py   --train-data data/train.txt   --val-data data/val.txt   --tokenizer-name gpt2   --seq-len 1024   --batch-size 8   --num-steps 200
Training started at 2025-11-28 17:06:33
Using device: cuda
Arguments: Namespace(vocab_size=10000, max_position_embeddings=10000, n_layers=2, n_heads=4, d_model=128, d_ff=512, dropout=0.1, use_tensor_parallel=False, batch_size=8, seq_len=1024, num_steps=200, lr=0.0003, no_amp=False, checkpoint_path='checkpoints/minigpt_single.pt', resume=False, train_data=['data/train.txt'], val_data=['data/val.txt'], tokenizer_name='gpt2', use_toy_data=False)
train dataset size: 32
val dataset size: 46
Starting from scratch
('step 10 / 200 ', 'train loss: 10.6139 ', 'val loss: 10.5547 ', 'val ppl: 38357.6676 ', 'amp: True')
('step 20 / 200 ', 'train loss: 9.9051 ', 'val loss: 9.8221 ', 'val ppl: 18436.6312 ', 'amp: True')
('step 30 / 200 ', 'train loss: 8.9364 ', 'val loss: 8.9257 ', 'val ppl: 7523.1618 ', 'amp: True')
('step 40 / 200 ', 'train loss: 8.2415 ', 'val loss: 8.1917 ', 'val ppl: 3610.9226 ', 'amp: True')
('step 50 / 200 ', 'train loss: 7.6098 ', 'val loss: 7.5641 ', 'val ppl: 1927.7076 ', 'amp: True')
('step 60 / 200 ', 'train loss: 6.9796 ', 'val loss: 7.0511 ', 'val ppl: 1154.1602 ', 'amp: True')
('step 70 / 200 ', 'train loss: 6.5684 ', 'val loss: 6.6589 ', 'val ppl: 779.6782 ', 'amp: True')
('step 80 / 200 ', 'train loss: 6.3557 ', 'val loss: 6.3737 ', 'val ppl: 586.2495 ', 'amp: True')
('step 90 / 200 ', 'train loss: 6.0814 ', 'val loss: 6.1744 ', 'val ppl: 480.2882 ', 'amp: True')
('step 100 / 200 ', 'train loss: 5.8957 ', 'val loss: 6.0485 ', 'val ppl: 423.4948 ', 'amp: True')
('step 110 / 200 ', 'train loss: 5.7787 ', 'val loss: 5.9599 ', 'val ppl: 387.5877 ', 'amp: True')
('step 120 / 200 ', 'train loss: 5.6908 ', 'val loss: 5.8859 ', 'val ppl: 359.9240 ', 'amp: True')
('step 130 / 200 ', 'train loss: 5.5336 ', 'val loss: 5.8183 ', 'val ppl: 336.4132 ', 'amp: True')
('step 140 / 200 ', 'train loss: 5.4770 ', 'val loss: 5.7541 ', 'val ppl: 315.4703 ', 'amp: True')
('step 150 / 200 ', 'train loss: 5.3506 ', 'val loss: 5.6913 ', 'val ppl: 296.2860 ', 'amp: True')
('step 160 / 200 ', 'train loss: 5.3421 ', 'val loss: 5.6343 ', 'val ppl: 279.8512 ', 'amp: True')
('step 170 / 200 ', 'train loss: 5.2963 ', 'val loss: 5.5781 ', 'val ppl: 264.5793 ', 'amp: True')
('step 180 / 200 ', 'train loss: 5.1676 ', 'val loss: 5.5264 ', 'val ppl: 251.2452 ', 'amp: True')
('step 190 / 200 ', 'train loss: 5.1330 ', 'val loss: 5.4757 ', 'val ppl: 238.8231 ', 'amp: True')
('step 200 / 200 ', 'train loss: 5.3526 ', 'val loss: 5.4268 ', 'val ppl: 227.4290 ', 'amp: True')
```

分布式数据并行训练：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py   --train-data data/train.txt   --val-data data/val.txt   --tokenizer-name gpt2   --seq-len 1024   --batch-size 8   --num-steps 200
W1128 17:13:01.472000 765093 site-packages/torch/distributed/run.py:792] 
W1128 17:13:01.472000 765093 site-packages/torch/distributed/run.py:792] *****************************************
W1128 17:13:01.472000 765093 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1128 17:13:01.472000 765093 site-packages/torch/distributed/run.py:792] *****************************************
Training started at 2025-11-28 17:13:02
[rank 0] Using device: cuda:0
Arguments: Namespace(vocab_size=10000, max_position_embeddings=10000, n_layers=2, n_heads=4, d_model=128, d_ff=512, dropout=0.1, use_tensor_parallel=False, batch_size=8, seq_len=1024, num_steps=200, lr=0.0003, no_amp=False, checkpoint_path='checkpoints/minigpt_ddp.pt', resume=False, train_data=['data/train.txt'], val_data=['data/val.txt'], tokenizer_name='gpt2', use_toy_data=False)
[rank 0] Starting from scratch
('step 10 / 200 ', 'train loss: 10.6083 ', 'val loss: 10.5442 ', 'val ppl: 37956.3182 ', 'amp: True')
('step 20 / 200 ', 'train loss: 9.8831 ', 'val loss: 9.7849 ', 'val ppl: 17764.3321 ', 'amp: True')
('step 30 / 200 ', 'train loss: 8.9969 ', 'val loss: 8.8916 ', 'val ppl: 7270.7905 ', 'amp: True')
('step 40 / 200 ', 'train loss: 8.2000 ', 'val loss: 8.1592 ', 'val ppl: 3495.2977 ', 'amp: True')
('step 50 / 200 ', 'train loss: 7.5706 ', 'val loss: 7.5283 ', 'val ppl: 1859.9687 ', 'amp: True')
('step 60 / 200 ', 'train loss: 6.9359 ', 'val loss: 7.0018 ', 'val ppl: 1098.5724 ', 'amp: True')
('step 70 / 200 ', 'train loss: 6.5049 ', 'val loss: 6.6018 ', 'val ppl: 736.3947 ', 'amp: True')
('step 80 / 200 ', 'train loss: 6.1959 ', 'val loss: 6.3190 ', 'val ppl: 555.0407 ', 'amp: True')
('step 90 / 200 ', 'train loss: 5.9893 ', 'val loss: 6.1319 ', 'val ppl: 460.3293 ', 'amp: True')
('step 100 / 200 ', 'train loss: 5.8101 ', 'val loss: 6.0096 ', 'val ppl: 407.3259 ', 'amp: True')
('step 110 / 200 ', 'train loss: 5.7503 ', 'val loss: 5.9232 ', 'val ppl: 373.6082 ', 'amp: True')
('step 120 / 200 ', 'train loss: 5.6801 ', 'val loss: 5.8452 ', 'val ppl: 345.5622 ', 'amp: True')
('step 130 / 200 ', 'train loss: 5.5241 ', 'val loss: 5.7663 ', 'val ppl: 319.3581 ', 'amp: True')
('step 140 / 200 ', 'train loss: 5.3991 ', 'val loss: 5.6918 ', 'val ppl: 296.4208 ', 'amp: True')
('step 150 / 200 ', 'train loss: 5.2615 ', 'val loss: 5.6235 ', 'val ppl: 276.8460 ', 'amp: True')
('step 160 / 200 ', 'train loss: 5.2492 ', 'val loss: 5.5583 ', 'val ppl: 259.3840 ', 'amp: True')
('step 170 / 200 ', 'train loss: 5.2032 ', 'val loss: 5.4943 ', 'val ppl: 243.2923 ', 'amp: True')
('step 180 / 200 ', 'train loss: 5.1544 ', 'val loss: 5.4326 ', 'val ppl: 228.7322 ', 'amp: True')
('step 190 / 200 ', 'train loss: 5.1739 ', 'val loss: 5.3719 ', 'val ppl: 215.2690 ', 'amp: True')
('step 200 / 200 ', 'train loss: 5.0410 ', 'val loss: 5.3140 ', 'val ppl: 203.1597 ', 'amp: True')
Training finished.
```
