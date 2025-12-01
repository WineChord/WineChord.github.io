---
classes: wide2
title: "从零实现 LLM Training：006. Row Parallel"
excerpt: "在已有 Column Parallel 基础上实现 Row Parallel Linear。"
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

我们已经实现并集成了 Column Parallel Linear，下面我们来做 Row Parallel Linear。

他本质上是对参数矩阵做按行切分，那么就需要输入按列切分。

## `tensor_parallel.py`

```python
class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("dist is not initialized")
        tp_group = get_tensor_parallel_group()
        tp_world_size = dist.get_world_size(tp_group)
        if in_features % tp_world_size != 0:
            raise ValueError("in_features must be divisible by tp_world_size")
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size
        self.in_per_rank = in_features // tp_world_size
        self.rank = dist.get_rank(tp_group)
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
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
        if self.input_is_parallel:
            x_local = x
        else:
            if x.size(-1) != self.in_features:
                raise ValueError("x.size(-1) must be equal to in_features")
            start = self.rank * self.in_per_rank
            end = start + self.in_per_rank
            x_local = x[..., start:end]
        y_local = torch.matmul(x_local, self.weight.t())
        dist.all_reduce(y_local, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            y_local = y_local + self.bias
        return y_local

```

这里有一些细节：

* 存在 `input_is_parallel` 的选项，并且通常来说，他前面就是 Column Parallel，这个 flag 基本上就是 true，这个 flag 可以和 Column Parallel 那边的 `gather_output` 看成是对称的。
* `weight` 的形状是先 output dimension 后 input dimension，这是为了和 pytorch Linear 内部的写法对齐，并且 nn.init.xx 各种都是默认参数是先 output dimension 后 input dimension 的，然后 forward 的时候就要注意是 `x @ w.t()` 有个转置。
* 对于 Row Parallel 来说，矩阵乘法之后是先做 all-reduce，再加上 bias，而对于 Column Parallel 来说，矩阵乘法之后是先加 bias，再做 all-gather

## `test_row_parallel_linear.py`

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from tensor_parallel import RowParallelLinear, init_tensor_parallel


def setup_distributed() -> torch.device:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device


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
    out_features = 6
    if in_features % world_size != 0:
        raise RuntimeError("in_features must be divisible by world_size")
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    ref_linear = nn.Linear(in_features, out_features, bias=True).to(device)
    tp_linear = RowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        input_is_parallel=True,
    ).to(device)
    with torch.no_grad():
        in_per_rank = in_features // world_size
        start = rank * in_per_rank
        end = start + in_per_rank
        tp_linear.weight.copy_(ref_linear.weight[:, start:end])
        if tp_linear.bias is not None:
            tp_linear.bias.copy_(ref_linear.bias[:])
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    x_full = torch.randn(batch_size, in_features, device=device)
    in_per_rank = in_features // world_size
    start = rank * in_per_rank
    end = start + in_per_rank
    x_local = x_full[..., start:end]
    y_ref = ref_linear(x_full)
    y_tp = tp_linear(x_local)
    diff = (y_ref - y_tp).abs().max()
    diff_val = diff.item()
    if rank == 0:
        print("max |y_ref - y_tp| = ", diff_val)
    cleanup_distributed()


if __name__ == "__main__":
    main()

```

运行结果如下：

```shell
$ torchrun --nproc-per-node=2 test_row_parallel_linear.py 
W1126 21:11:43.144000 3314148 site-packages/torch/distributed/run.py:792] 
W1126 21:11:43.144000 3314148 site-packages/torch/distributed/run.py:792] *****************************************
W1126 21:11:43.144000 3314148 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 21:11:43.144000 3314148 site-packages/torch/distributed/run.py:792] *****************************************
world_size = 2
max |y_ref - y_tp| =  5.960464477539063e-08
```

这里我们会发现 Row Parallel 是会引入微小误差的，因为原来是大矩阵乘法，现在被切成了小矩阵乘法附带 all-reduce，但是 Column Parallel 其实是没有误差的，因为他的结果是做拼接，和单卡的乘加顺序、次数是完全一致的。
