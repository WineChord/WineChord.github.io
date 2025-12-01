---
classes: wide2
title: "从零实现 LLM Training：018. LR Cosine Scheduler"
excerpt: "为学习率引入 cosine scheduler，并将调度状态写入 checkpoint。"
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

这个 PR 为我们现在的 learning rate 加上一个 scheduler，之前我们的 learning rate 每次训练是固定的 3e-4，然后后面 resume 再去降 learning rate，使用不是很友好，我们可以使用 cosine learning rate，并把 scheduler 相关信息存到 checkpoint 中，这样当 resume 的时候，当时使用的 learning rate 也可以一并 resume。

核心式子其实就是：

```python
def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    warmup_steps: int,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
```



## `checkpoint.py`

![image-20251130000205325](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251130000205325.png)

## `train_ddp.py`

![image-20251130000324216](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251130000324216.png)

![image-20251130000346107](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251130000346107.png)

![image-20251130000407915](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251130000407915.png)

train_minimal.py 的改法类似。

## 运行

还是跑我们的 train_ddp.py，这时候把使用的语料总 token 数加大，然后用上 cosine lr scheduler，并且带一个 linear warmup，可以看到 lr 的变化符合预期：

```shell
('epoch 1 step 10 / 10000 ', 'lr: 0.000033 ', 'train loss: 9.8186 ', 'val loss: 10.0169 ', 'val ppl: 22402.7661 ', 'amp: True')
('epoch 1 step 20 / 10000 ', 'lr: 0.000063 ', 'train loss: 9.4491 ', 'val loss: 8.9739 ', 'val ppl: 7894.2767 ', 'amp: True')
('epoch 1 step 30 / 10000 ', 'lr: 0.000093 ', 'train loss: 8.9832 ', 'val loss: 8.3670 ', 'val ppl: 4302.6351 ', 'amp: True')
('epoch 1 step 40 / 10000 ', 'lr: 0.000123 ', 'train loss: 8.4849 ', 'val loss: 7.7705 ', 'val ppl: 2369.6728 ', 'amp: True')
('epoch 1 step 50 / 10000 ', 'lr: 0.000153 ', 'train loss: 7.8802 ', 'val loss: 7.3438 ', 'val ppl: 1546.6266 ', 'amp: True')
('epoch 1 step 60 / 10000 ', 'lr: 0.000183 ', 'train loss: 6.9094 ', 'val loss: 7.1296 ', 'val ppl: 1248.3786 ', 'amp: True')
('epoch 1 step 70 / 10000 ', 'lr: 0.000213 ', 'train loss: 7.4544 ', 'val loss: 7.0697 ', 'val ppl: 1175.8176 ', 'amp: True')
('epoch 1 step 80 / 10000 ', 'lr: 0.000243 ', 'train loss: 6.6120 ', 'val loss: 7.0894 ', 'val ppl: 1199.1414 ', 'amp: True')
('epoch 1 step 90 / 10000 ', 'lr: 0.000273 ', 'train loss: 7.7091 ', 'val loss: 6.9887 ', 'val ppl: 1084.3237 ', 'amp: True')
('epoch 1 step 100 / 10000 ', 'lr: 0.000300 ', 'train loss: 7.5927 ', 'val loss: 6.9006 ', 'val ppl: 992.8808 ', 'amp: True')
('epoch 1 step 110 / 10000 ', 'lr: 0.000300 ', 'train loss: 4.6842 ', 'val loss: 6.8607 ', 'val ppl: 954.0262 ', 'amp: True')
('epoch 1 step 120 / 10000 ', 'lr: 0.000300 ', 'train loss: 7.1439 ', 'val loss: 6.7731 ', 'val ppl: 874.0214 ', 'amp: True')
('epoch 1 step 130 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.5979 ', 'val loss: 6.6851 ', 'val ppl: 800.3592 ', 'amp: True')
('epoch 1 step 140 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.0690 ', 'val loss: 6.6097 ', 'val ppl: 742.2547 ', 'amp: True')
('epoch 1 step 150 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.4554 ', 'val loss: 6.5700 ', 'val ppl: 713.3606 ', 'amp: True')
('epoch 1 step 160 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.5241 ', 'val loss: 6.5210 ', 'val ppl: 679.2363 ', 'amp: True')
('epoch 1 step 170 / 10000 ', 'lr: 0.000300 ', 'train loss: 7.0583 ', 'val loss: 6.5053 ', 'val ppl: 668.6759 ', 'amp: True')
('epoch 1 step 180 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.8849 ', 'val loss: 6.4673 ', 'val ppl: 643.7444 ', 'amp: True')
('epoch 1 step 190 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.4060 ', 'val loss: 6.4323 ', 'val ppl: 621.6098 ', 'amp: True')
('epoch 1 step 200 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.9052 ', 'val loss: 6.3986 ', 'val ppl: 601.0060 ', 'amp: True')
('epoch 1 step 210 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.4502 ', 'val loss: 6.3733 ', 'val ppl: 585.9680 ', 'amp: True')
('epoch 1 step 220 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.2548 ', 'val loss: 6.3450 ', 'val ppl: 569.6192 ', 'amp: True')
('epoch 1 step 230 / 10000 ', 'lr: 0.000300 ', 'train loss: 5.5311 ', 'val loss: 6.3404 ', 'val ppl: 567.0251 ', 'amp: True')
('epoch 1 step 240 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.2708 ', 'val loss: 6.3115 ', 'val ppl: 550.8576 ', 'amp: True')
('epoch 1 step 250 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.3344 ', 'val loss: 6.2927 ', 'val ppl: 540.6181 ', 'amp: True')
('epoch 1 step 260 / 10000 ', 'lr: 0.000300 ', 'train loss: 7.0485 ', 'val loss: 6.2473 ', 'val ppl: 516.6214 ', 'amp: True')
('epoch 1 step 270 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.3838 ', 'val loss: 6.2316 ', 'val ppl: 508.5623 ', 'amp: True')
('epoch 1 step 280 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.8108 ', 'val loss: 6.2109 ', 'val ppl: 498.1735 ', 'amp: True')
('epoch 1 step 290 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.0598 ', 'val loss: 6.1749 ', 'val ppl: 480.5564 ', 'amp: True')
('epoch 1 step 300 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.4197 ', 'val loss: 6.2051 ', 'val ppl: 495.2453 ', 'amp: True')
('epoch 1 step 310 / 10000 ', 'lr: 0.000300 ', 'train loss: 5.9679 ', 'val loss: 6.1879 ', 'val ppl: 486.8039 ', 'amp: True')
('epoch 1 step 320 / 10000 ', 'lr: 0.000300 ', 'train loss: 5.7620 ', 'val loss: 6.1917 ', 'val ppl: 488.6696 ', 'amp: True')
('epoch 1 step 330 / 10000 ', 'lr: 0.000300 ', 'train loss: 5.3066 ', 'val loss: 6.1527 ', 'val ppl: 469.9994 ', 'amp: True')
('epoch 1 step 340 / 10000 ', 'lr: 0.000300 ', 'train loss: 6.3324 ', 'val loss: 6.1243 ', 'val ppl: 456.8097 ', 'amp: True')
('epoch 1 step 350 / 10000 ', 'lr: 0.000300 ', 'train loss: 5.9847 ', 'val loss: 6.1094 ', 'val ppl: 450.0807 ', 'amp: True')
('epoch 1 step 360 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.6618 ', 'val loss: 6.0828 ', 'val ppl: 438.2571 ', 'amp: True')
('epoch 1 step 370 / 10000 ', 'lr: 0.000299 ', 'train loss: 5.7799 ', 'val loss: 6.0715 ', 'val ppl: 433.3214 ', 'amp: True')
('epoch 1 step 380 / 10000 ', 'lr: 0.000299 ', 'train loss: 5.3388 ', 'val loss: 6.0769 ', 'val ppl: 435.6569 ', 'amp: True')
('epoch 1 step 390 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.4828 ', 'val loss: 6.0445 ', 'val ppl: 421.7905 ', 'amp: True')
('epoch 1 step 400 / 10000 ', 'lr: 0.000299 ', 'train loss: 4.1484 ', 'val loss: 6.0126 ', 'val ppl: 408.5635 ', 'amp: True')
('epoch 1 step 410 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.6390 ', 'val loss: 6.0436 ', 'val ppl: 421.3955 ', 'amp: True')
('epoch 1 step 420 / 10000 ', 'lr: 0.000299 ', 'train loss: 5.1276 ', 'val loss: 5.9919 ', 'val ppl: 400.1583 ', 'amp: True')
('epoch 1 step 430 / 10000 ', 'lr: 0.000299 ', 'train loss: 5.8495 ', 'val loss: 5.9965 ', 'val ppl: 402.0034 ', 'amp: True')
('epoch 1 step 440 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.1805 ', 'val loss: 5.9720 ', 'val ppl: 392.3036 ', 'amp: True')
('epoch 1 step 450 / 10000 ', 'lr: 0.000299 ', 'train loss: 4.4069 ', 'val loss: 5.9693 ', 'val ppl: 391.2352 ', 'amp: True')
('epoch 1 step 460 / 10000 ', 'lr: 0.000299 ', 'train loss: 4.4153 ', 'val loss: 5.9644 ', 'val ppl: 389.3242 ', 'amp: True')
('epoch 1 step 470 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.3935 ', 'val loss: 5.9341 ', 'val ppl: 377.6950 ', 'amp: True')
('epoch 1 step 480 / 10000 ', 'lr: 0.000299 ', 'train loss: 4.7016 ', 'val loss: 5.9138 ', 'val ppl: 370.1100 ', 'amp: True')
('epoch 1 step 490 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.2590 ', 'val loss: 5.9157 ', 'val ppl: 370.8168 ', 'amp: True')
('epoch 1 step 500 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.1099 ', 'val loss: 5.9099 ', 'val ppl: 368.6788 ', 'amp: True')
('epoch 1 step 510 / 10000 ', 'lr: 0.000299 ', 'train loss: 5.5213 ', 'val loss: 5.9108 ', 'val ppl: 369.0025 ', 'amp: True')
('epoch 1 step 520 / 10000 ', 'lr: 0.000299 ', 'train loss: 6.1863 ', 'val loss: 5.9253 ', 'val ppl: 374.3995 ', 'amp: True')
('epoch 1 step 530 / 10000 ', 'lr: 0.000299 ', 'train loss: 4.6636 ', 'val loss: 5.8958 ', 'val ppl: 363.4895 ', 'amp: True')
('epoch 1 step 540 / 10000 ', 'lr: 0.000299 ', 'train loss: 5.9903 ', 'val loss: 5.8953 ', 'val ppl: 363.3412 ', 'amp: True')
('epoch 1 step 550 / 10000 ', 'lr: 0.000298 ', 'train loss: 6.4008 ', 'val loss: 5.8664 ', 'val ppl: 352.9644 ', 'amp: True')
('epoch 1 step 560 / 10000 ', 'lr: 0.000298 ', 'train loss: 5.9950 ', 'val loss: 5.8558 ', 'val ppl: 349.2584 ', 'amp: True')
('epoch 1 step 570 / 10000 ', 'lr: 0.000298 ', 'train loss: 6.5759 ', 'val loss: 5.8520 ', 'val ppl: 347.9365 ', 'amp: True')
('epoch 1 step 580 / 10000 ', 'lr: 0.000298 ', 'train loss: 6.0202 ', 'val loss: 5.8478 ', 'val ppl: 346.4661 ', 'amp: True')
('epoch 1 step 590 / 100
```
