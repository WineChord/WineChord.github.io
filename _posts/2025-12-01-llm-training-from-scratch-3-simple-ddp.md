---
classes: wide2
title: "从零实现 LLM Training：3. Simple DDP"
excerpt: "使用 PyTorch DDP 实现最简单的数据并行训练。"
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

在实现了 mini-GPT 以及简单的 training loop 之后，我们可以上一些最简单的分布式训练，比如数据并行。数据并行本质上就是每张卡跑相同的模型，但是使用不同的数据，每张卡各自 forward backward 之后通过 all-reduce 来得到平均梯度。

为了简单起见，这里我们直接使用 pytorch 的数据并行，只需要额外添加一个 `train_ddp.py` 文件即可。

## `train_ddp.py`

```python
import os

import torch
import torch.distributed as dist
from config import GPTConfig
from model import GPTModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(local_rank: int) -> bool:
    return local_rank == 0


def main():
    device, local_rank = setup_distributed()
    if is_main_process(local_rank):
        print(f"[rank {local_rank}] Using device: {device}")
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
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )
    dataset = ToyRandomDataset(
        vocab_size=config.vocab_size,
        seq_len=32,
        num_samples=1000,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    ddp_model.train()
    num_steps = 50
    step = 0
    for epoch in range(1, 1000):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            step += 1
            if step > num_steps:
                break
            input_ids = batch["input_ids"].to(device)  # [B, T]
            labels = batch["labels"].to(device)  # [B, T]
            attention_mask = batch["attention_mask"].to(device)  # [B, T]
            optimizer.zero_grad()
            logits, loss = ddp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss.backward()
            optimizer.step()
            if is_main_process(local_rank) and step % 10 == 0:
                print(f"[step {step} / {num_steps}] loss = {loss.item():.4f}")
        if step > num_steps:
            break
    if is_main_process(local_rank):
        print("Training finished.")
    cleanup_distributed()


if __name__ == "__main__":
    main()

```

这里我们依然是使用一个随机的 Toy Dataset，每次生成一个长度为 seq_len 的 id 列表，然后 labels 暂时等于生成的 id 列表（为了简单起见）。

在 `setup_distributed` 的时候我们需要初始化分布式的环境，需要调用 `dist.init_process_group(backend="nccl")`，然后我们需要用 `torchrun` 来运行这个代码，`torchrun` 会自动给每张卡注入对应的 `LOCAL_RANK` 环境变量，所以我们可以通过 `int(os.environ["LOCAL_RANK"])` 拿到对应的 local rank。

在构造完模型后，需要用 `DDP` 给包装一下，从而使用到数据并行的能力，实际的效果是每个 rank 拿一份自己的 model，放到对应的 GPU 上，`DDP` 则负责在 `loss.backward()` 的时候，让每个 rank 各自算梯度，用 `all_reduce` 把梯度求平均，让所有参数保持同步。

在构造数据集时，需要额外构造一个 `DistributedSampler` 来传给 `DataLoader`，他会把整个 dataset 切成 world size 份，每个 rank 只看到其中的一份。防止不同 GPU 重复使用同一批样本。其中还有一句 `sampler.set_epoch(epoch)` 是为了在每个 epoch 打乱时采用不同的随机种子。

剩下的逻辑就和之前单卡训练非常相似了。

## 运行

DDP 的运行需要 `torchrun`，命令如下：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py 
W1126 10:45:37.836000 2760123 site-packages/torch/distributed/run.py:792] 
W1126 10:45:37.836000 2760123 site-packages/torch/distributed/run.py:792] *****************************************
W1126 10:45:37.836000 2760123 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 10:45:37.836000 2760123 site-packages/torch/distributed/run.py:792] *****************************************
[rank 0] Using device: cuda:0
[step 10 / 50] loss = 9.3437
[step 20 / 50] loss = 9.3408
[step 30 / 50] loss = 9.3145
[step 40 / 50] loss = 9.3891
[step 50 / 50] loss = 9.3715
Training finished.
```

其中 `--nproc-per-node=2` 表示每个节点有两张卡，然后默认情况下我们是一个节点（一台机器）。

这个执行之后，会启动两个进程（rank0, rank1），并给进程注入环境变量：

* `RANK` 表示全局 GPU 索引（多机多卡里每卡唯一）
* `WORLD_SIZE` 表示总共有多少张卡
* `LOCAL_RANK` 表示他是本机的第几张卡
