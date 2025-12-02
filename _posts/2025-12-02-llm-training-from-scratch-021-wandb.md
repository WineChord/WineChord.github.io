---
classes: wide2
title: "从零实现 LLM Training：021. WandB"
excerpt: "使用 WandB 记录训练过程，方便后续分析。"
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

在折腾了全训练流程，并初步分析了 pytorch profiler trace 之后，是时候把我们的日志和训练信息上报到远端平台方便分析查看了，避免每次都是在本地看日志文件。

此处我们选择久负盛名的 wandb 进行接入，首先要执行 `pip install wandb`，然后 `wandb login`，当然，你要先注册一个账号。

## 代码变更

加一些 import：

![image-20251202111656821](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111656821.png)

构造 wandb 配置：

![image-20251202111720937](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111720937.png)

上报 log：

![image-20251202111737805](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111737805.png)

命令行选项：

![image-20251202111752350](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111752350.png)

train_ddp.py 的改动类似：

![image-20251202111829719](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111829719.png)

![image-20251202111845443](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111845443.png)

![image-20251202111902659](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111902659.png)

![image-20251202111914454](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202111914454.png)



## 运行

我们可以运行一下，前后可以看到很 fancy 的 wandb 日志输出（以 ddp 为例）：

```shell
$ torchrun --nproc-per-node=2 train_ddp.py \
  --train-data data/train.txt \
  --tokenizer-name gpt2 \
  --seq-len 512 \
  --batch-size 2 \
  --num-steps 50 \
  --use-wandb \
  --wandb-project mini-llm \
  --wandb-run-name ddp_test
W1202 11:11:33.874000 2211647 site-packages/torch/distributed/run.py:792] 
W1202 11:11:33.874000 2211647 site-packages/torch/distributed/run.py:792] *****************************************
W1202 11:11:33.874000 2211647 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1202 11:11:33.874000 2211647 site-packages/torch/distributed/run.py:792] *****************************************
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Tracking run with wandb version 0.19.8
wandb: Run data is saved locally in /data/projects/rosellm/rosellm/rosetrainer/wandb/run-20251202_111136-mrkum49w
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ddp_test
[2025-12-02 11:11:37] Training started at 2025-12-02 11:11:37
[2025-12-02 11:11:37] [rank 0] Using device: cuda:0
[2025-12-02 11:11:37] Arguments: Namespace(vocab_size=10000, max_position_embeddings=10000, n_layers=2, n_heads=4, d_model=128, d_ff=512, dropout=0.1, use_tensor_parallel=False, use_activation_checkpoint=False, batch_size=2, seq_len=512, num_steps=50, lr=0.0003, no_amp=False, checkpoint_path='checkpoints/minigpt_ddp.pt', resume=False, lr_scheduler='cosine', warmup_steps=100, use_profiler=False, train_data=['data/train.txt'], val_data=[], val_ratio=0.1, tokenizer_name='gpt2', use_toy_data=False, max_tokens=None, data_seed=None, use_wandb=True, wandb_project='mini-llm', wandb_run_name='ddp_test')
total files: 1total files: 1
total tokens: 937660

total tokens: 937660
[2025-12-02 11:11:40] train dataset size: 1648
[2025-12-02 11:11:40] val dataset size: 183
[2025-12-02 11:11:40] steps per epoch: 412
[2025-12-02 11:11:40] [rank 0] Starting from scratch
[2025-12-02 11:11:42] ('epoch 1 step 10 / 50 ', 'lr: 0.000033 ', 'step time: 0.16', 'toks/s (per rank): 6537.89', 'train loss: 10.9411 ', 'val loss: 10.9447 ', 'val ppl: 56653.4597 ', 'dt: 1.98s ', 'eta: 0.00h ', 'amp: True')
[2025-12-02 11:11:44] ('epoch 1 step 20 / 50 ', 'lr: 0.000063 ', 'step time: 0.16', 'toks/s (per rank): 6560.64', 'train loss: 10.8948 ', 'val loss: 10.8905 ', 'val ppl: 53663.6150 ', 'dt: 1.93s ', 'eta: 0.00h ', 'amp: True')
[2025-12-02 11:11:46] ('epoch 1 step 30 / 50 ', 'lr: 0.000093 ', 'step time: 0.16', 'toks/s (per rank): 6546.46', 'train loss: 10.8261 ', 'val loss: 10.7966 ', 'val ppl: 48855.7236 ', 'dt: 1.70s ', 'eta: 0.00h ', 'amp: True')
[2025-12-02 11:11:47] ('epoch 1 step 40 / 50 ', 'lr: 0.000123 ', 'step time: 0.16', 'toks/s (per rank): 6591.09', 'train loss: 10.7083 ', 'val loss: 10.6405 ', 'val ppl: 41792.1065 ', 'dt: 1.88s ', 'eta: 0.00h ', 'amp: True')
[2025-12-02 11:11:49] ('epoch 1 step 50 / 50 ', 'lr: 0.000153 ', 'step time: 0.16', 'toks/s (per rank): 6565.52', 'train loss: 10.5141 ', 'val loss: 10.3555 ', 'val ppl: 31430.3487 ', 'dt: 1.70s ', 'eta: 0.00h ', 'amp: True')
[2025-12-02 11:11:49] Training finished.
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                     amp ▁▁▁▁▁
wandb:   global_tokens_per_sec ▁▄▂█▅
wandb:                      lr ▁▃▅▆█
wandb: tokens_per_sec_per_rank ▁▄▂█▅
wandb:              train/loss █▇▆▄▁
wandb:                val/loss █▇▆▄▁
wandb:                 val/ppl █▇▆▄▁
wandb: 
wandb: Run summary:
wandb:                     amp 1
wandb:   global_tokens_per_sec 13131.04329
wandb:                      lr 0.00015
wandb: tokens_per_sec_per_rank 6565.52165
wandb:              train/loss 10.51408
wandb:                val/loss 10.35553
wandb:                 val/ppl 31430.34874
wandb: 
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20251202_111136-mrkum49w/logs
```

然后我们就可以在网站上看到 fancy 的上报了，还是很 fancy 的：

![image-20251202112138094](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251202112138094.png)
