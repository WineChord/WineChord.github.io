---
classes: wide2
title: "从零实现 LLM Training：012. Argparse"
excerpt: "为训练脚本引入 argparse 命令行参数，向工业级实现迈进。"
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

我们已经实现了基础的张量并行、混合精度训练、checkpoint 等，为了走向工业级别的实现，是时候给他加上命令行选项了！

## `train_minimal.py`

```diff
diff --git a/rosellm/rosetrainer/train_minimal.py b/rosellm/rosetrainer/train_minimal.py
index 12fa1fc..338ec3b 100644
--- a/rosellm/rosetrainer/train_minimal.py
+++ b/rosellm/rosetrainer/train_minimal.py
@@ -1,3 +1,4 @@
+import argparse
 import os
 
 import torch
@@ -33,36 +34,36 @@ class ToyRandomDataset(Dataset):
         }
 
 
-def main():
+def main(args: argparse.Namespace) -> None:
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print("Using device:", device)
-    checkpoint_path = "checkpoints/minigpt_single.pt"
-    resume = False
+    checkpoint_path = args.checkpoint_path
+    resume = args.resume
     config = GPTConfig(
-        vocab_size=10000,
-        max_position_embeddings=128,
-        n_layers=2,
-        n_heads=4,
-        d_model=128,
-        d_ff=512,
-        dropout=0.1,
+        vocab_size=args.vocab_size,
+        max_position_embeddings=args.max_position_embeddings,
+        n_layers=args.n_layers,
+        n_heads=args.n_heads,
+        d_model=args.d_model,
+        d_ff=args.d_ff,
+        dropout=args.dropout,
     )
     model = GPTModel(config).to(device)
     dataset = ToyRandomDataset(
         vocab_size=config.vocab_size,
-        seq_len=32,
+        seq_len=args.seq_len,
         num_samples=1000,
     )
     dataloader = DataLoader(
         dataset,
-        batch_size=8,
+        batch_size=args.batch_size,
         shuffle=True,
     )
-    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
-    use_amp = device.type == "cuda"
+    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
+    use_amp = device.type == "cuda" and not args.no_amp
     scaler = GradScaler(enabled=use_amp)
     model.train()
-    num_steps = 50
+    num_steps = args.num_steps
     step = 0
     if resume and os.path.exists(checkpoint_path):
         print(f"Resuming from checkpoint {checkpoint_path}")
@@ -115,5 +116,98 @@ def main():
             )
 
 
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="Train minimal GPT model.")
+    parser.add_argument(
+        "--vocab-size",
+        type=int,
+        default=10000,
+        help="Vocabulary size.",
+    )
+    parser.add_argument(
+        "--max-position-embeddings",
+        type=int,
+        default=128,
+        help="Max sequence length.",
+    )
+    parser.add_argument(
+        "--n-layers",
+        type=int,
+        default=2,
+        help="Number of Transformer layers.",
+    )
+    parser.add_argument(
+        "--n-heads",
+        type=int,
+        default=4,
+        help="Number of attention heads.",
+    )
+    parser.add_argument(
+        "--d-model",
+        type=int,
+        default=128,
+        help="Model hidden size.",
+    )
+    parser.add_argument(
+        "--d-ff",
+        type=int,
+        default=512,
+        help="FFN hidden size.",
+    )
+    parser.add_argument(
+        "--dropout",
+        type=float,
+        default=0.1,
+        help="Dropout probability.",
+    )
+    parser.add_argument(
+        "--use-tensor-parallel",
+        action="store_true",
+        help="Enable tensor parallel blocks.",
+    )
+    parser.add_argument(
+        "--batch-size",
+        type=int,
+        default=8,
+        help="Batch size per step.",
+    )
+    parser.add_argument(
+        "--seq-len",
+        type=int,
+        default=32,
+        help="Sequence length.",
+    )
+    parser.add_argument(
+        "--num-steps",
+        type=int,
+        default=50,
+        help="Number of training steps.",
+    )
+    parser.add_argument(
+        "--lr",
+        type=float,
+        default=3e-4,
+        help="Learning rate.",
+    )
+    parser.add_argument(
+        "--no-amp",
+        action="store_true",
+        help="Disable AMP even on CUDA.",
+    )
+    parser.add_argument(
+        "--checkpoint-path",
+        type=str,
+        default="checkpoints/minigpt_single.pt",
+        help="Path to checkpoint file.",
+    )
+    parser.add_argument(
+        "--resume",
+        action="store_true",
+        help="Resume training from checkpoint.",
+    )
+    return parser.parse_args()
+
+
 if __name__ == "__main__":
-    main()
+    args = parse_args()
+    main(args)
```

## `train_ddp.py`

```diff
diff --git a/rosellm/rosetrainer/train_ddp.py b/rosellm/rosetrainer/train_ddp.py
index 47c2b68..0c3ad7f 100644
--- a/rosellm/rosetrainer/train_ddp.py
+++ b/rosellm/rosetrainer/train_ddp.py
@@ -1,3 +1,4 @@
+import argparse
 import os
 
 import torch
@@ -51,21 +52,21 @@ def is_main_process(local_rank: int) -> bool:
     return local_rank == 0
 
 
-def main():
+def main(args: argparse.Namespace) -> None:
     device, local_rank = setup_distributed()
-    checkpoint_path = "checkpoints/minigpt_ddp.pt"
-    resume = False
+    checkpoint_path = args.checkpoint_path
+    resume = args.resume
     if is_main_process(local_rank):
         print(f"[rank {local_rank}] Using device: {device}")
         os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
     config = GPTConfig(
-        vocab_size=10000,
-        max_position_embeddings=128,
-        n_layers=2,
-        n_heads=4,
-        d_model=128,
-        d_ff=512,
-        dropout=0.1,
+        vocab_size=args.vocab_size,
+        max_position_embeddings=args.max_position_embeddings,
+        n_layers=args.n_layers,
+        n_heads=args.n_heads,
+        d_model=args.d_model,
+        d_ff=args.d_ff,
+        dropout=args.dropout,
     )
     model = GPTModel(config).to(device)
     ddp_model = DDP(
@@ -76,7 +77,7 @@ def main():
     )
     dataset = ToyRandomDataset(
         vocab_size=config.vocab_size,
-        seq_len=32,
+        seq_len=args.seq_len,
         num_samples=1000,
     )
     sampler = DistributedSampler(
@@ -87,14 +88,14 @@ def main():
     )
     dataloader = DataLoader(
         dataset,
-        batch_size=8,
+        batch_size=args.batch_size,
         sampler=sampler,
     )
-    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
+    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
     use_amp = device.type == "cuda"
     scaler = GradScaler(enabled=use_amp)
     ddp_model.train()
-    num_steps = 50
+    num_steps = args.num_steps
     step = 0
     if resume and os.path.exists(checkpoint_path):
         print(f"[rank {local_rank}] Resuming from checkpoint {checkpoint_path}")
@@ -162,5 +163,98 @@ def main():
     cleanup_distributed()
     
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="DDP training for GPT model.")
+    parser.add_argument(
+        "--vocab-size",
+        type=int,
+        default=10000,
+        help="Vocabulary size.",
+    )
+    parser.add_argument(
+        "--max-position-embeddings",
+        type=int,
+        default=128,
+        help="Max sequence length.",
+    )
+    parser.add_argument(
+        "--n-layers",
+        type=int,
+        default=2,
+        help="Number of Transformer layers.",
+    )
+    parser.add_argument(
+        "--n-heads",
+        type=int,
+        default=4,
+        help="Number of attention heads.",
+    )
+    parser.add_argument(
+        "--d-model",
+        type=int,
+        default=128,
+        help="Model hidden size.",
+    )
+    parser.add_argument(
+        "--d-ff",
+        type=int,
+        default=512,
+        help="FFN hidden size.",
+    )
+    parser.add_argument(
+        "--dropout",
+        type=float,
+        default=0.1,
+        help="Dropout probability.",
+    )
+    parser.add_argument(
+        "--use-tensor-parallel",
+        action="store_true",
+        help="Enable tensor parallel blocks.",
+    )
+    parser.add_argument(
+        "--batch-size",
+        type=int,
+        default=8,
+        help="Batch size per rank.",
+    )
+    parser.add_argument(
+        "--seq-len",
+        type=int,
+        default=32,
+        help="Sequence length.",
+    )
+    parser.add_argument(
+        "--num-steps",
+        type=int,
+        default=50,
+        help="Total training steps.",
+    )
+    parser.add_argument(
+        "--lr",
+        type=float,
+        default=3e-4,
+        help="Learning rate.",
+    )
+    parser.add_argument(
+        "--no-amp",
+        action="store_true",
+        help="Disable AMP even on CUDA.",
+    )
+    parser.add_argument(
+        "--checkpoint-path",
+        type=str,
+        default="checkpoints/minigpt_ddp.pt",
+        help="Path to checkpoint file.",
+    )
+    parser.add_argument(
+        "--resume",
+        action="store_true",
+        help="Resume training from checkpoint.",
+    )
+    return parser.parse_args()
+
+
 if __name__ == "__main__":
-    main()
+    args = parse_args()
+    main(args)
```





## 运行

```shell
$ python train_minimal.py \
  --n-layers 4 \
  --d-model 256 \
  --n-heads 4 \
  --d-ff 1024 \
  --batch-size 16 \
  --seq-len 64 \
  --num-steps 100 \
  --use-tensor-parallel \
  --resume \
  --checkpoint-path checkpoints/exp1.pt
Using device: cuda
Resume flag is set, but checkpoint not found. Starting from scratch.
step 10 / 100  loss: 9.3688  amp: True
step 20 / 100  loss: 9.3830  amp: True
step 30 / 100  loss: 9.3733  amp: True
step 40 / 100  loss: 9.3916  amp: True
step 50 / 100  loss: 9.3693  amp: True
step 60 / 100  loss: 9.3690  amp: True
```

第二次运行：

```shell
$ python train_minimal.py   --n-layers 4   --d-model 256   --n-heads 4   --d-ff 1024   --batch-size 16   --seq-len 64   --num-steps 100   --use-tensor-parallel   --resume   --checkpoint-path checkpoints/exp1.pt
Using device: cuda
Resuming from checkpoint checkpoints/exp1.pt
Resumed from step 60
step 70 / 100  loss: 9.3858  amp: True
step 80 / 100  loss: 9.3328  amp: True
step 90 / 100  loss: 9.3256  amp: True
step 100 / 100  loss: 9.3718  amp: True
```

然后运行 ddp：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py \
  --n-layers 4 \
  --d-model 256 \
  --n-heads 4 \
  --d-ff 1024 \
  --seq-len 64 \
  --batch-size 16 \
  --num-steps 200 \
  --checkpoint-path checkpoints/exp_ddp.pt \
  --resume
W1127 20:16:16.653000 220625 site-packages/torch/distributed/run.py:792] 
W1127 20:16:16.653000 220625 site-packages/torch/distributed/run.py:792] *****************************************
W1127 20:16:16.653000 220625 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 20:16:16.653000 220625 site-packages/torch/distributed/run.py:792] *****************************************
[rank 0] Using device: cuda:0
[rank 0] Resume flag is set, but checkpoint not found. Starting from scratch.
[step 10 / 200]  loss = 9.4041  amp = True
[step 20 / 200]  loss = 9.3670  amp = True
[step 30 / 200]  loss = 9.3738  amp = True
[step 40 / 200]  loss = 9.3965  amp = True
[step 50 / 200]  loss = 9.3431  amp = True
[step 60 / 200]  loss = 9.3778  amp = True
[step 70 / 200]  loss = 9.3556  amp = True
[step 80 / 200]  loss = 9.3475  amp = True
[step 90 / 200]  loss = 9.3084  amp = True
[step 100 / 200]  loss = 9.3308  amp = True
[step 110 / 200]  loss = 9.2820  amp = True
[step 120 / 200]  loss = 9.2637  amp = True
[step 130 / 200]  loss = 9.2769  amp = True
[step 140 / 200]  loss = 9.2609  amp = True
[step 150 / 200]  loss = 9.2423  amp = True
[step 160 / 200]  loss = 9.2251  amp = True
[step 170 / 200]  loss = 9.2467  amp = True
[step 180 / 200]  loss = 9.2489  amp = True
[step 190 / 200]  loss = 9.2630  amp = True
[step 200 / 200]  loss = 9.2392  amp = True
Training finished.
```

第二次运行：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py \
  --n-layers 4 \
  --d-model 256 \
  --n-heads 4 \
  --d-ff 1024 \
  --seq-len 64 \
  --batch-size 16 \
  --num-steps 200 \
  --checkpoint-path checkpoints/exp_ddp.pt \
  --resume
W1127 20:16:27.440000 220758 site-packages/torch/distributed/run.py:792] 
W1127 20:16:27.440000 220758 site-packages/torch/distributed/run.py:792] *****************************************
W1127 20:16:27.440000 220758 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 20:16:27.440000 220758 site-packages/torch/distributed/run.py:792] *****************************************
[rank 0] Using device: cuda:0
[rank 1] Resuming from checkpoint checkpoints/exp_ddp.pt
[rank 0] Resuming from checkpoint checkpoints/exp_ddp.pt
[rank 0] Resumed from step 200
Training finished.
[rank 1] Resumed from step 200
```
