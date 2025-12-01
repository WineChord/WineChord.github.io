---
classes: wide2
title: "从零实现 LLM Training：011. Checkpoints"
excerpt: "为训练过程加入 checkpoint 容错机制，支持从中间状态恢复。"
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

我们已经依次做了张量并行，混合精度训练，下面我们来做一些容错，也就是 checkpoint，在训练过程中定期保存 checkpoint，这样在失败的时候可以从中间某个 checkpoint 加载然后继续训练。

为了简单起见，这个 PR 在 DDP 时仅在 rank0 上存完整的 checkpoint，成熟的实现应该在每张卡上去存 shard。

## `checkpoint.py`

```python
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def _maybe_serialize_config(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    return config


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    scaler: Optional["torch.amp.GradScaler"] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    map_location: Optional[str] = None,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    if map_location is None:
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("step", 0))
    extra = ckpt.get("extra")
    return step, extra

```

基本上就是存模型的 state_dict 以及 optimizer states，grad scaler 这些东西。

## `train_minimal.py`

```diff
diff --git a/rosellm/rosetrainer/train_minimal.py b/rosellm/rosetrainer/train_minimal.py
index 3395fcb..12fa1fc 100644
--- a/rosellm/rosetrainer/train_minimal.py
+++ b/rosellm/rosetrainer/train_minimal.py
@@ -1,4 +1,7 @@
+import os
+
 import torch
+from checkpoint import load_checkpoint, save_checkpoint
 from config import GPTConfig
 from model import GPTModel
 from torch.amp import GradScaler, autocast
@@ -33,6 +36,8 @@ class ToyRandomDataset(Dataset):
 def main():
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print("Using device:", device)
+    checkpoint_path = "checkpoints/minigpt_single.pt"
+    resume = False
     config = GPTConfig(
         vocab_size=10000,
         max_position_embeddings=128,
@@ -59,6 +64,14 @@ def main():
     model.train()
     num_steps = 50
     step = 0
+    if resume and os.path.exists(checkpoint_path):
+        print(f"Resuming from checkpoint {checkpoint_path}")
+        step, extra = load_checkpoint(checkpoint_path, model, optimizer, scaler)
+        print(f"Resumed from step {step}")
+    elif resume:
+        print("Resume flag is set, but checkpoint not found. Starting from scratch.")
+    else:
+        print("Starting from scratch")
     for batch in dataloader:
         step += 1
         if step > num_steps:
@@ -85,6 +98,15 @@ def main():
             )
             loss.backward()
             optimizer.step()
+        if step % 20 == 0:
+            save_checkpoint(
+                checkpoint_path,
+                model=model,
+                optimizer=optimizer,
+                step=step,
+                scaler=scaler if use_amp else None,
+                extra={"note": "single_gpt_minimal"},
+            )
         if step % 10 == 0:
             print(
                 f"step {step} / {num_steps} ",
```



## `train_ddp.py`

```diff
diff --git a/rosellm/rosetrainer/train_ddp.py b/rosellm/rosetrainer/train_ddp.py
index 506c135..47c2b68 100644
--- a/rosellm/rosetrainer/train_ddp.py
+++ b/rosellm/rosetrainer/train_ddp.py
@@ -2,6 +2,7 @@ import os
 
 import torch
 import torch.distributed as dist
+from checkpoint import load_checkpoint, save_checkpoint
 from config import GPTConfig
 from model import GPTModel
 from torch.amp import GradScaler, autocast
@@ -52,8 +53,11 @@ def is_main_process(local_rank: int) -> bool:
 
 def main():
     device, local_rank = setup_distributed()
+    checkpoint_path = "checkpoints/minigpt_ddp.pt"
+    resume = False
     if is_main_process(local_rank):
         print(f"[rank {local_rank}] Using device: {device}")
+        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
     config = GPTConfig(
         vocab_size=10000,
         max_position_embeddings=128,
@@ -92,6 +96,22 @@ def main():
     ddp_model.train()
     num_steps = 50
     step = 0
+    if resume and os.path.exists(checkpoint_path):
+        print(f"[rank {local_rank}] Resuming from checkpoint {checkpoint_path}")
+        step, extra = load_checkpoint(
+            checkpoint_path,
+            ddp_model.module,
+            optimizer,
+            scaler,
+            map_location=device.type,
+        )
+        print(f"[rank {local_rank}] Resumed from step {step}")
+    elif resume and is_main_process(local_rank):
+        print(
+            f"[rank {local_rank}] Resume flag is set, but checkpoint not found. Starting from scratch."
+        )
+    elif is_main_process(local_rank):
+        print(f"[rank {local_rank}] Starting from scratch")
     for epoch in range(1, 1000):
         sampler.set_epoch(epoch)
         for batch in dataloader:
@@ -120,6 +140,15 @@ def main():
                 )
                 loss.backward()
                 optimizer.step()
+            if is_main_process(local_rank) and step % 20 == 0:
+                save_checkpoint(
+                    checkpoint_path,
+                    model=ddp_model.module,
+                    optimizer=optimizer,
+                    step=step,
+                    scaler=scaler if use_amp else None,
+                    extra={"note": "minigpt_ddp"},
+                )
             if is_main_process(local_rank) and step % 10 == 0:
                 print(
                     f"[step {step} / {num_steps}] ",
```



## 运行

```shell
$ python train_minimal.py 
Using device: cuda
Starting from scratch
step 10 / 50  loss: 9.4214  amp: True
step 20 / 50  loss: 9.3472  amp: True
step 30 / 50  loss: 9.3074  amp: True
step 40 / 50  loss: 9.3985  amp: True
step 50 / 50  loss: 9.3647  amp: True
```

把 resume 在代码里手动改为 true：

```shell
$ python train_minimal.py 
Using device: cuda
Resuming from checkpoint checkpoints/minigpt_single.pt
Resumed from step 40
step 50 / 50  loss: 9.3564  amp: True
```

然后运行 ddp：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py 
W1127 19:33:08.185000 202022 site-packages/torch/distributed/run.py:792] 
W1127 19:33:08.185000 202022 site-packages/torch/distributed/run.py:792] *****************************************
W1127 19:33:08.185000 202022 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 19:33:08.185000 202022 site-packages/torch/distributed/run.py:792] *****************************************
[rank 0] Using device: cuda:0
[rank 0] Starting from scratch
[step 10 / 50]  loss = 9.3744  amp = True
[step 20 / 50]  loss = 9.3875  amp = True
[step 30 / 50]  loss = 9.3425  amp = True
[step 40 / 50]  loss = 9.3408  amp = True
[step 50 / 50]  loss = 9.3457  amp = True
Training finished.
```

手动把代码里的 resume 改成 true：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py 
W1127 19:33:26.301000 202200 site-packages/torch/distributed/run.py:792] 
W1127 19:33:26.301000 202200 site-packages/torch/distributed/run.py:792] *****************************************
W1127 19:33:26.301000 202200 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 19:33:26.301000 202200 site-packages/torch/distributed/run.py:792] *****************************************
[rank 0] Using device: cuda:0
[rank 1] Resuming from checkpoint checkpoints/minigpt_ddp.pt
[rank 0] Resuming from checkpoint checkpoints/minigpt_ddp.pt
[rank 0] Resumed from step 40
[rank 1] Resumed from step 40
[step 50 / 50]  loss = 9.3251  amp = True
Training finished.
```
