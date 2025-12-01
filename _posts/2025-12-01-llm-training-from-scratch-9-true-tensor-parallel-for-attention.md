---
classes: wide2
title: "从零实现 LLM Training：9. True Tensor Parallel for Attention"
excerpt: "通过按 head 维度切分 QKV，让 Attention 形成真正的张量并行。"
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

# 从零实现 LLM Training：9. True Tensor Parallel for Attention

上一个 PR 我们给 Attention layer 加上了 Row Parallel Linear，但是实际上和 QKV 所使用的 Column Parallel Linear 没有形成搭配，会造成额外的 all-gather 开销，在本文对应 PR 中，我们将 QKV 的结果自然按 head 做切分，从而避免额外的 all-gather 开销。

## `model.py`

```diff
diff --git a/rosellm/rosetrainer/model.py b/rosellm/rosetrainer/model.py
index 102e308..56a2054 100644
--- a/rosellm/rosetrainer/model.py
+++ b/rosellm/rosetrainer/model.py
@@ -19,22 +19,30 @@ class MultiHeadSelfAttention(nn.Module):
         self.d_model = config.d_model
         self.n_heads = config.n_heads
         self.d_head = config.d_model // config.n_heads
-        use_tp = getattr(config, "use_tensor_parallel", False)
-        if use_tp and dist.is_available() and dist.is_initialized():
+        use_tp_cfg = getattr(config, "use_tensor_parallel", False)
+        self.use_tp = use_tp_cfg and dist.is_available() and dist.is_initialized()
+        if self.use_tp:
             init_tensor_parallel()
+            tp_world_size = dist.get_world_size()
+            if self.n_heads % tp_world_size != 0:
+                raise ValueError("n_heads must be divisible by tp_world_size")
+            self.tp_world_size = tp_world_size
+            self.local_heads = self.n_heads // tp_world_size
             self.qkv_proj = ColumnParallelLinear(
                 in_features=config.d_model,
                 out_features=3 * config.d_model,
                 bias=True,
-                gather_output=True,
+                gather_output=False,
             )
             self.out_proj = RowParallelLinear(
                 in_features=config.d_model,
                 out_features=config.d_model,
                 bias=True,
-                input_is_parallel=False,
+                input_is_parallel=True,
             )
         else:
+            self.tp_world_size = 1
+            self.local_heads = self.n_heads
             self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
             self.out_proj = nn.Linear(config.d_model, config.d_model)
         self.dropout = nn.Dropout(config.dropout)
@@ -57,7 +65,7 @@ class MultiHeadSelfAttention(nn.Module):
     ):
         bsz, seq_len, _ = x.size()
         qkv = self.qkv_proj(x)
-        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.d_head)
+        qkv = qkv.view(bsz, seq_len, 3, self.local_heads, self.d_head)
         qkv = qkv.permute(2, 0, 3, 1, 4)
         q, k, v = qkv[0], qkv[1], qkv[2]
         attn_scores = q @ k.transpose(-2, -1) * self.d_head**-0.5
@@ -70,7 +78,7 @@ class MultiHeadSelfAttention(nn.Module):
         attn_weights = self.dropout(attn_weights)
         attn_output = attn_weights @ v
         attn_output = attn_output.transpose(1, 2).contiguous()
-        attn_output = attn_output.view(bsz, seq_len, self.d_model)
+        attn_output = attn_output.view(bsz, seq_len, self.local_heads * self.d_head)
         out = self.out_proj(attn_output)
         out = self.dropout(out)
         return out
```

主要修改点就是添加了 `local_heads` 以及把 gather_output 改成了 false，把 input_is_parallel 改成了 true。

## `test_attention_tp_vs_dense.py`

```diff
diff --git a/rosellm/rosetrainer/test_attention_tp_vs_dense.py b/rosellm/rosetrainer/test_attention_tp_vs_dense.py
index fd3b63b..24e21d7 100644
--- a/rosellm/rosetrainer/test_attention_tp_vs_dense.py
+++ b/rosellm/rosetrainer/test_attention_tp_vs_dense.py
@@ -55,13 +55,41 @@ def copy_qkv_from_dense_to_tp(
     with torch.no_grad():
         linear_dense: nn.Linear = attn_dense.qkv_proj
         col_tp = attn_tp.qkv_proj
-        out_features = linear_dense.out_features
-        out_per_rank = out_features // world_size
-        start = rank * out_per_rank
-        end = start + out_per_rank
-        col_tp.weight.copy_(linear_dense.weight[start:end, :])
+
+        d_model = attn_dense.d_model
+        n_heads = attn_dense.n_heads
+        d_head = attn_dense.d_head
+        assert d_model == n_heads * d_head
+
+        local_heads = n_heads // world_size
+        local_dim = local_heads * d_head
+        head_start = rank * local_heads
+        head_end = head_start + local_heads
+
+        q_offset = 0
+        k_offset = d_model
+        v_offset = 2 * d_model
+
+        q_start = q_offset + head_start * d_head
+        q_end = q_offset + head_end * d_head
+        k_start = k_offset + head_start * d_head
+        k_end = k_offset + head_end * d_head
+        v_start = v_offset + head_start * d_head
+        v_end = v_offset + head_end * d_head
+
+        q_weight = linear_dense.weight[q_start:q_end, :]
+        k_weight = linear_dense.weight[k_start:k_end, :]
+        v_weight = linear_dense.weight[v_start:v_end, :]
+
+        col_tp.weight[:local_dim, :].copy_(q_weight)
+        col_tp.weight[local_dim : 2 * local_dim, :].copy_(k_weight)
+        col_tp.weight[2 * local_dim : 3 * local_dim, :].copy_(v_weight)
+
         if col_tp.bias is not None:
-            col_tp.bias.copy_(linear_dense.bias[start:end])
+            q_bias = linear_dense.bias[q_start:q_end]
+            k_bias = linear_dense.bias[k_start:k_end]
+            v_bias = linear_dense.bias[v_start:v_end]
+            col_tp.bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))
```

相应需要稍微修改一下测试文件，运行如下：

```shell
$ torchrun --nproc-per-node=2 test_attention_tp_vs_dense.py 
W1127 14:25:50.446000 72319 site-packages/torch/distributed/run.py:792] 
W1127 14:25:50.446000 72319 site-packages/torch/distributed/run.py:792] *****************************************
W1127 14:25:50.446000 72319 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1127 14:25:50.446000 72319 site-packages/torch/distributed/run.py:792] *****************************************
world_size = 2
y_dense shape: torch.Size([2, 8, 64])
y_tp shape: torch.Size([2, 8, 64])
max |y_dense - y_tp| =  2.384185791015625e-07
```
