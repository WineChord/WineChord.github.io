---
classes: wide2
title: "从零实现 LLM Training：20. PyTorch Profiler and NVTX"
excerpt: "使用 PyTorch profiler 与 NVTX 捕捉 trace，深入分析训练性能瓶颈。"
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

在实现了基础的模型训练，以及 activation checkpointing 之后，我们通过日志以及 nvidia-smi 可以大概看出来开了一些功能之后，耗时以及显存的变化，但是对整个程序运行的认知其实还不够彻底，此时我们需要引入 pytorch profiler 以及 nvtx 等工具来捕捉 trace，以更好地理解程序的行为。

## profile 对比

### 开启 AMP vs 不开启 AMP

先看 profiler step 的时间：

![image-20251201194255308](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251201194255308.png)

![image-20251201194313508](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/image-20251201194313508.png)

开启的时候是 106ms，不开启的时候是 185ms
