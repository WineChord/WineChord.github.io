---
classes: wide2
title: "从零实现 LLM Training：4. Column Parallel"
excerpt: "引入列张量并行（Column Parallel）作为张量并行的第一步。"
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

在实现了 mini-GPT，简单的 train loop，较简单的数据并行之后，我们这次可以正式开始最基础的张量并行。

张量并行分为列张量并行和行张量并行，其中列张量并行表示对权重矩阵按列切分，相当于同样的激活值在不同的卡上过模型按列切分的不同部分，得到的结果需要 all-gather 成完整的激活值，在后面直接搭配行张量并行时，这个 all-gather 会省略掉，因为行张量并行恰好需要切分后的激活值，行张量并行的结果是完整的激活值形状，但是每张卡有不同的具体激活值，需要进行 all-reduce 来拿到统一的结果。

本文的新 PR 就先实现最简单的列张量并行。

## `tensor_parallel.py`

```python
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

_TP_GROUP: Optional[dist.ProcessGroup] = None


def init_tensor_parallel(tp_size: Optional[int] = None) -> None:
    global _TP_GROUP
    if not dist.is_initialized():
        raise RuntimeError("dist is not initialized")
    world_size = dist.get_world_size()
    if tp_size is None:
        tp_size = world_size
    if tp_size != world_size:
        raise NotImplementedError("currently we only support tp_size == world_size")
    _TP_GROUP = dist.group.WORLD


def get_tensor_parallel_group() -> dist.ProcessGroup:
    if _TP_GROUP is None:
        raise RuntimeError("tensor parallel group is not initialized")
    return _TP_GROUP


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("dist is not initialized")
        tp_group = get_tensor_parallel_group()
        tp_world_size = dist.get_world_size(tp_group)
        if out_features % tp_world_size != 0:
            raise ValueError("out_features must be divisible by tp_world_size")
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size
        self.out_per_rank = out_features // tp_world_size
        self.rank = dist.get_rank(tp_group)
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_local = y_local + self.bias
        if not self.gather_output or self.tp_world_size == 1:
            return y_local
        out_list = [torch.empty_like(y_local) for _ in range(self.tp_world_size)]
        dist.all_gather(out_list, y_local, group=self.tp_group)
        y = torch.cat(out_list, dim=-1)
        return y

```

这里我们定义了一个全局变量 `_TP_GROUP`，表示张量并行的通信组（process group），并在初始化的时候暂时赋值他为 `dist.group.WORLD`，相当于我们默认全组都是张量并行组，后续的 PR 中我们会做深入的细化。

然后我们实现了 `ColumnParallelLinear` ，在 init 中将 weight 按照 rank 进行切分，在前向中，手动调用了 all-gather 来聚合出完整的结果。

接下来我们需要一个小的测试来验证我们的实现是正确的。

## `test_tensor_parallel_linear.py`

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from tensor_parallel import ColumnParallelLinear, init_tensor_parallel


def setup_distributed() -> torch.device:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def main() -> None:
    device = setup_distributed()
    init_tensor_parallel()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        print(f"world_size = {world_size}")
    batch_size = 4
    in_features = 8
    out_features = 12
    if out_features % world_size != 0:
        raise RuntimeError("out_features must be divisible by world_size")
    torch.manual_seed(42)
    ref_linear = nn.Linear(in_features, out_features, bias=True).to(device)
    tp_linear = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        gather_output=True,
    ).to(device)
    with torch.no_grad():
        out_per_rank = out_features // world_size
        start = rank * out_per_rank
        end = start + out_per_rank
        tp_linear.weight.copy_(ref_linear.weight[start:end, :])
        tp_linear.bias.copy_(ref_linear.bias[start:end])
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, device=device)
    y_ref = ref_linear(x)
    y_tp = tp_linear(x)
    diff = (y_ref - y_tp).abs().max()
    diff_val = diff.item()
    if rank == 0:
        print("max |y_ref - t_tp| = ", diff_val)
    cleanup_distributed()


if __name__ == "__main__":
    main()

```

这个测试验证了经过列并行 linear 和经过普通的 linear 得到的结果完全一致，具体运行如下：

```shell
$ torchrun --nproc-per-node=2 test_tensor_parallel_linear.py 
W1126 16:57:28.144000 3084989 site-packages/torch/distributed/run.py:792] 
W1126 16:57:28.144000 3084989 site-packages/torch/distributed/run.py:792] *****************************************
W1126 16:57:28.144000 3084989 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 16:57:28.144000 3084989 site-packages/torch/distributed/run.py:792] *****************************************
world_size = 2
max |y_ref - t_tp| =  0.0
```
