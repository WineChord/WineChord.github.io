---
classes: wide2
title: "从零实现 LLM Training：013. Eval and Logging"
excerpt: "加入验证集评估和日志记录，用 loss 与 PPL 监控训练效果。"
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

在实现了张量并行，混合精度，checkpoint，argparse 之后，我们可以再引入一些正儿八经但是没有那么激动的技术，比如 eval 和 logging。

对于 eval 来说，实际上就是划分训练集和验证集，通过观察验证集上的 loss 以及 PPL（perplexity，困惑度），来客观评价当前模型学习的效果。

PPL 可以展开解释一下：

- 语言模型里一般用的 loss 是「每个 token 的平均 **negative log-likelihood**」：
  $$
  \text{loss} = -\frac{1}{N}\sum_{i=1}^N \log p(w_i \mid \text{context}_i)
  $$
  
- Perplexity 的定义是：
  $$
  \text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log p(w_i \mid \text{context}_i)\right)
  $$
  
- 你可以看到，括号里面这坨，正好就是上面的 loss，所以：
  $$
  \text{PPL} = \exp(\text{loss})
  $$

他本质实际上就是每个 token 平均 loss 再取一个 exp，再感性一点的理解是：“模型平均有多少个候选词在竞争”，比如假如模型是在词表中所有词均匀选，那么平均 loss 其实就是 $-\log(1/V)$，再取 exp 刚好就是 $V$，然后这个指标和 loss 一样，是越小越好，并且 PPL 比 loss 观察的粒度会更大一点，毕竟去了一个 exp，你观察 loss 对比 2.6 2.8 感觉都差不太多，但是对应的 PPL 还是有些更大的差距的。

logging 部分暂时也是简单做，就搞一个写到文件的功能就好了，并且只在 rank0 写。

## `train_minimal.py`

```diff
diff --git a/rosellm/rosetrainer/train_minimal.py b/rosellm/rosetrainer/train_minimal.py
index 338ec3b..e3bc5d9 100644
--- a/rosellm/rosetrainer/train_minimal.py
+++ b/rosellm/rosetrainer/train_minimal.py
@@ -1,5 +1,7 @@
 import argparse
+import math
 import os
+from datetime import datetime
 
 import torch
 from checkpoint import load_checkpoint, save_checkpoint
@@ -34,9 +36,56 @@ class ToyRandomDataset(Dataset):
         }
 
 
+def log_line(path: str, text: str) -> None:
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+    with open(path, "a", encoding="utf-8") as f:
+        f.write(str(text) + "\n")
+
+
+def evaluate(
+    model: GPTModel,
+    dataloader: DataLoader,
+    device: torch.device,
+    use_amp: bool,
+) -> float:
+    model_was_training = model.training
+    model.eval()
+    total_loss = 0.0
+    total_tokens = 0
+    with torch.no_grad():
+        for batch in dataloader:
+            input_ids = batch["input_ids"].to(device)
+            labels = batch["labels"].to(device)
+            attention_mask = batch["attention_mask"].to(device)
+            if use_amp:
+                with autocast(device_type=device.type):
+                    _, loss = model(
+                        input_ids=input_ids,
+                        attention_mask=attention_mask,
+                        labels=labels,
+                    )
+            else:
+                _, loss = model(
+                    input_ids=input_ids,
+                    attention_mask=attention_mask,
+                    labels=labels,
+                )
+            batch_tokens = labels.numel()
+            total_loss += float(loss.item()) * batch_tokens
+            total_tokens += batch_tokens
+    avg_loss = total_loss / max(total_tokens, 1)
+    if model_was_training:
+        model.train()
+    return avg_loss
+
+
 def main(args: argparse.Namespace) -> None:
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-    print("Using device:", device)
+    log_path = "logs/train_minimal.log"
+    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
+    log_line(log_path, f"Training started at {timestamp}")
+    log_line(log_path, f"Using device: {device}")
+    log_line(log_path, f"Arguments: {args}")
     checkpoint_path = args.checkpoint_path
     resume = args.resume
     config = GPTConfig(
@@ -49,16 +98,27 @@ def main(args: argparse.Namespace) -> None:
         dropout=args.dropout,
     )
     model = GPTModel(config).to(device)
-    dataset = ToyRandomDataset(
+    full_dataset = ToyRandomDataset(
         vocab_size=config.vocab_size,
         seq_len=args.seq_len,
         num_samples=1000,
     )
-    dataloader = DataLoader(
-        dataset,
+    val_size = max(int(0.2 * len(full_dataset)), 1)
+    train_size = len(full_dataset) - val_size
+    train_dataset, val_dataset = torch.utils.data.random_split(
+        full_dataset,
+        [train_size, val_size],
+    )
+    train_dataloader = DataLoader(
+        train_dataset,
         batch_size=args.batch_size,
         shuffle=True,
     )
+    val_dataloader = DataLoader(
+        val_dataset,
+        batch_size=args.batch_size,
+        shuffle=False,
+    )
     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
     use_amp = device.type == "cuda" and not args.no_amp
     scaler = GradScaler(enabled=use_amp)
@@ -73,7 +133,7 @@ def main(args: argparse.Namespace) -> None:
         print("Resume flag is set, but checkpoint not found. Starting from scratch.")
     else:
         print("Starting from scratch")
-    for batch in dataloader:
+    for batch in train_dataloader:
         step += 1
         if step > num_steps:
             break
@@ -109,11 +169,22 @@ def main(args: argparse.Namespace) -> None:
                 extra={"note": "single_gpt_minimal"},
             )
         if step % 10 == 0:
-            print(
+            val_loss = evaluate(
+                model,
+                val_dataloader,
+                device=device,
+                use_amp=use_amp,
+            )
+            val_ppl = math.exp(val_loss)
+            msg = (
                 f"step {step} / {num_steps} ",
-                f"loss: {loss.item():.4f} ",
+                f"train loss: {loss.item():.4f} ",
+                f"val loss: {val_loss:.4f} ",
+                f"val ppl: {val_ppl:.4f} ",
                 f"amp: {use_amp}",
             )
+            print(msg)
+            log_line(log_path, msg)
 
 
 def parse_args() -> argparse.Namespace:
```



## `train_ddp.py`

```diff
diff --git a/rosellm/rosetrainer/train_ddp.py b/rosellm/rosetrainer/train_ddp.py
index 0c3ad7f..8373362 100644
--- a/rosellm/rosetrainer/train_ddp.py
+++ b/rosellm/rosetrainer/train_ddp.py
@@ -1,5 +1,7 @@
 import argparse
+import math
 import os
+from datetime import datetime
 
 import torch
 import torch.distributed as dist
@@ -36,6 +38,57 @@ class ToyRandomDataset(Dataset):
         }
 
 
+def log_line(path: str, text: str) -> None:
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+    with open(path, "a", encoding="utf-8") as f:
+        f.write(str(text) + "\n")
+
+
+def evaluate_ddp(
+    ddp_model: DDP,
+    dataloader: DataLoader,
+    device: torch.device,
+    use_amp: bool,
+) -> float:
+    model_was_training = ddp_model.module.training
+    ddp_model.eval()
+    total_loss = 0.0
+    total_tokens = 0
+    with torch.no_grad():
+        for batch in dataloader:
+            input_ids = batch["input_ids"].to(device)
+            labels = batch["labels"].to(device)
+            attention_mask = batch["attention_mask"].to(device)
+            if use_amp:
+                with autocast(device_type=device.type):
+                    _, loss = ddp_model(
+                        input_ids=input_ids,
+                        attention_mask=attention_mask,
+                        labels=labels,
+                    )
+            else:
+                _, loss = ddp_model(
+                    input_ids=input_ids,
+                    attention_mask=attention_mask,
+                    labels=labels,
+                )
+            batch_tokens = labels.numel()
+            total_loss += float(loss.item()) * batch_tokens
+            total_tokens += batch_tokens
+    loss_tensor = torch.tensor(
+        [total_loss, total_tokens],
+        dtype=torch.float64,
+        device=device,
+    )
+    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
+    total_loss_all = float(loss_tensor[0].item())
+    total_tokens_all = float(loss_tensor[1].item())
+    avg_loss = total_loss_all / max(total_tokens_all, 1.0)
+    if model_was_training:
+        ddp_model.module.train()
+    return avg_loss
+
+
 def setup_distributed():
     dist.init_process_group(backend="nccl")
     local_rank = int(os.environ["LOCAL_RANK"])
@@ -56,8 +109,12 @@ def main(args: argparse.Namespace) -> None:
     device, local_rank = setup_distributed()
     checkpoint_path = args.checkpoint_path
     resume = args.resume
+    log_path = "logs/train_ddp.log"
     if is_main_process(local_rank):
-        print(f"[rank {local_rank}] Using device: {device}")
+        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
+        log_line(log_path, f"Training started at {timestamp}")
+        log_line(log_path, f"[rank {local_rank}] Using device: {device}")
+        log_line(log_path, f"Arguments: {args}")
         os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
     config = GPTConfig(
         vocab_size=args.vocab_size,
@@ -75,21 +132,39 @@ def main(args: argparse.Namespace) -> None:
         output_device=device.index,
         find_unused_parameters=False,
     )
-    dataset = ToyRandomDataset(
+    full_dataset = ToyRandomDataset(
         vocab_size=config.vocab_size,
         seq_len=args.seq_len,
         num_samples=1000,
     )
-    sampler = DistributedSampler(
-        dataset,
+    val_size = max(int(0.2 * len(full_dataset)), 1)
+    train_size = len(full_dataset) - val_size
+    train_dataset, val_dataset = torch.utils.data.random_split(
+        full_dataset,
+        [train_size, val_size],
+    )
+    train_sampler = DistributedSampler(
+        train_dataset,
         num_replicas=dist.get_world_size(),
         rank=dist.get_rank(),
         shuffle=True,
     )
-    dataloader = DataLoader(
-        dataset,
+    val_sampler = DistributedSampler(
+        val_dataset,
+        num_replicas=dist.get_world_size(),
+        rank=dist.get_rank(),
+        shuffle=False,
+    )
+    train_dataloader = DataLoader(
+        train_dataset,
+        batch_size=args.batch_size,
+        sampler=train_sampler,
+    )
+    val_dataloader = DataLoader(
+        val_dataset,
         batch_size=args.batch_size,
-        sampler=sampler,
+        sampler=val_sampler,
+        shuffle=False,
     )
     optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
     use_amp = device.type == "cuda"
@@ -114,8 +189,8 @@ def main(args: argparse.Namespace) -> None:
     elif is_main_process(local_rank):
         print(f"[rank {local_rank}] Starting from scratch")
     for epoch in range(1, 1000):
-        sampler.set_epoch(epoch)
-        for batch in dataloader:
+        train_sampler.set_epoch(epoch)
+        for batch in train_dataloader:
             step += 1
             if step > num_steps:
                 break
@@ -150,12 +225,24 @@ def main(args: argparse.Namespace) -> None:
                     scaler=scaler if use_amp else None,
                     extra={"note": "minigpt_ddp"},
                 )
-            if is_main_process(local_rank) and step % 10 == 0:
-                print(
-                    f"[step {step} / {num_steps}] ",
-                    f"loss = {loss.item():.4f} ",
-                    f"amp = {use_amp}",
+            if step % 10 == 0:
+                val_loss = evaluate_ddp(
+                    ddp_model,
+                    val_dataloader,
+                    device=device,
+                    use_amp=use_amp,
                 )
+                val_ppl = math.exp(val_loss)
+                if is_main_process(local_rank):
+                    msg = (
+                        f"step {step} / {num_steps} ",
+                        f"train loss: {loss.item():.4f} ",
+                        f"val loss: {val_loss:.4f} ",
+                        f"val ppl: {val_ppl:.4f} ",
+                        f"amp: {use_amp}",
+                    )
+                    print(msg)
+                    log_line(log_path, msg)
         if step > num_steps:
             break
     if is_main_process(local_rank):
```





## 运行

```shell
$ python train_minimal.py 
Starting from scratch
('step 10 / 50 ', 'train loss: 9.3665 ', 'val loss: 9.3725 ', 'val ppl: 11759.9524 ', 'amp: True')
('step 20 / 50 ', 'train loss: 9.3340 ', 'val loss: 9.3603 ', 'val ppl: 11617.5949 ', 'amp: True')
('step 30 / 50 ', 'train loss: 9.4282 ', 'val loss: 9.3626 ', 'val ppl: 11644.2892 ', 'amp: True')
('step 40 / 50 ', 'train loss: 9.3320 ', 'val loss: 9.3637 ', 'val ppl: 11657.9647 ', 'amp: True')
('step 50 / 50 ', 'train loss: 9.4083 ', 'val loss: 9.3562 ', 'val ppl: 11570.1428 ', 'amp: True')
```

```shell
$ torchrun --nproc-per-node=2 train_ddp.py --num-steps 30
W1127 21:25:03.335000 251381 site-packages/torch/distributed/run.py:792] 
W1127 21:25:03.335000 251381 site-packages/torch/distributed/run.py:792] *****************************************
W1127 21:25:03.335000 251381 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 21:25:03.335000 251381 site-packages/torch/distributed/run.py:792] *****************************************
[rank 0] Starting from scratch
('step 10 / 30 ', 'train loss: 9.3846 ', 'val loss: 9.3768 ', 'val ppl: 11811.5161 ', 'amp: True')
('step 20 / 30 ', 'train loss: 9.4233 ', 'val loss: 9.3639 ', 'val ppl: 11660.3308 ', 'amp: True')
('step 30 / 30 ', 'train loss: 9.3453 ', 'val loss: 9.3634 ', 'val ppl: 11653.6611 ', 'amp: True')
Training finished.
```
