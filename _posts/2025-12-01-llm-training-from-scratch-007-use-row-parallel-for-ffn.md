---
classes: wide2
title: "从零实现 LLM Training：007. Use Row Parallel for FFN"
excerpt: "把 Row Parallel Linear 引入 FFN 模块，配合 Column Parallel 完成张量并行。"
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

在实现了 Row Parallel Linear 之后，我们在新的 PR 中使用他，之前我们仅使用了 Column Parallel Linear，现在需要搭配 Row Parallel Linear 一块用，这样才组成完整的张量并行。

我们需要在 FFN 和 Attention 模块去引入 Row Parallel Linear，鉴于 Attention 模块较为复杂，本文新的 PR 将只改造 FFN 部分，以降低单 PR 引入的复杂度。

## `model.py`

```diff
diff --git a/rosellm/rosetrainer/model.py b/rosellm/rosetrainer/model.py
index fcc20d8..d95add6 100644
--- a/rosellm/rosetrainer/model.py
+++ b/rosellm/rosetrainer/model.py
@@ -5,7 +5,11 @@ import torch.distributed as dist
 import torch.nn as nn
 import torch.nn.functional as F
 from config import GPTConfig
-from tensor_parallel import ColumnParallelLinear, init_tensor_parallel
+from tensor_parallel import (
+    ColumnParallelLinear,
+    RowParallelLinear,
+    init_tensor_parallel,
+)
 
 
 class MultiHeadSelfAttention(nn.Module):
@@ -76,11 +80,17 @@ class FeedForward(nn.Module):
                 in_features=config.d_model,
                 out_features=config.d_ff,
                 bias=True,
-                gather_output=True,
+                gather_output=False,
+            )
+            self.fc2 = RowParallelLinear(
+                in_features=config.d_ff,
+                out_features=config.d_model,
+                bias=True,
+                input_is_parallel=True,
             )
         else:
             self.fc1 = nn.Linear(config.d_model, config.d_ff)
-        self.fc2 = nn.Linear(config.d_ff, config.d_model)
+            self.fc2 = nn.Linear(config.d_ff, config.d_model)
         self.dropout = nn.Dropout(config.dropout)
 
     def forward(self, x: torch.Tensor):
```

这是本 PR 需要修改的唯一的文件，其中我们在 FFN 使用了 Row Parallel Linear，他天然要求 input 是已经被切分了的，所以前面的 Column Parallel Linear 需要设置 gather_output 为 false，并设置他自己的 input_is_parallel 为 true。

至此本 PR 就结束了，我们可以通过运行之前的 `test_forward_tp.py` 来确定一切还都是正常的：

```shell
$ torchrun --nproc-per-node=2 test_forward_tp.py 
W1126 22:01:25.726000 3361578 site-packages/torch/distributed/run.py:792] 
W1126 22:01:25.726000 3361578 site-packages/torch/distributed/run.py:792] *****************************************
W1126 22:01:25.726000 3361578 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1126 22:01:25.726000 3361578 site-packages/torch/distributed/run.py:792] *****************************************
world_size: 2
max diff vs rank0: 0.0
max diff vs rank0: 0.0
input_ids shape: torch.Size([2, 16])
logits shape: torch.Size([2, 16, 10000])
loss: 9.294845581054688
```
