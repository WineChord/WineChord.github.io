---
classes: wide2
title: "从零实现 LLM Inference：077. 继续压 TTFT P99：离 TensorRT-LLM 还差 3ms"
excerpt: "076 把 gap 摆出来以后，我开始专门盯 online 的 TTFT 长尾。把 prefill chunk 调大、decode batch 调小、CUDA graph 预热、prefill 优先级、overlap 的 prefill completion 优先等一串看起来很“工程”的东西串起来之后，roseinfer 的 TTFT P99（scale=0.4）从 ~15ms 压到了 ~11.5ms，已经能明显压过 SGLang，但还是比 TensorRT-LLM 慢一截。后半篇记录两条我踩过的坑：+query decode 和 +single-chunk fp，它们都让 P99 变得更糟。"
categories:
  - LLM
  - Inference
tags:
  - LLM
  - Inference
toc: true
toc_sticky: true
mathjax: true
---

这一篇只盯一件事：**online 的 TTFT P99**。

我现在的心态已经很明确了：TPOT/ITL 这种 steady-state decode 指标，我们已经能压住 vLLM；真正恶心的还是 **TTFT 的长尾**，尤其在 `scale=0.4`（trace 被压缩到最重负载）的时候，一点小抖动都会被放大成尾巴。

---

## Benchmark 形态

沿用项目里的 serving benchmark：

- trace stage：回放真实 trace，`scale` 越小负载越大
- profile stage：只采最小样本的 torch/nsys，避免 profile 污染 benchmark 数字

四个指标：

- TTFT：time-to-first-token
- ITL：inter-token latency
- TPOT：time per output token
- E2E：end-to-end latency

顺手写一下关系式（输出 token 数为 $N$）：

$$
\mathrm{TPOT} = \frac{t_{end}-t_{first}}{N-1},\quad
\mathrm{E2E} = t_{end}-t_{start}
$$

---

## 当前最好结果

先把结论钉死：这一轮我手上最稳的配置，online 在 `scale=0.4` 下的 TTFT P99 已经能做到 **11.53ms**，但还是比 TensorRT-LLM 的 **8.03ms** 慢一截。

总览图：

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_compare.png)

只看 P99 / P90：

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_p90_only.png)

`scale=0.4` 的原始表格我直接贴在这里（避免“看图说话”）：

| backend | TTFT p90 | TTFT p99 | TPOT p90 | TPOT p99 | E2E p90 | E2E p99 |
|---|---:|---:|---:|---:|---:|---:|
| roseinfer | 10.33 | 11.53 | 1.44 | 1.56 | 186.91 | 205.67 |
| vLLM | 10.15 | 14.54 | 1.83 | 1.99 | 234.64 | 253.63 |
| SGLang | 9.39 | 14.89 | 1.21 | 1.35 | 156.65 | 170.35 |
| TensorRT-LLM | 6.30 | 8.03 | 1.44 | 1.90 | 187.42 | 193.03 |

<details>
<summary>online 原始数据（p50/p90/p99，全 scales）</summary>

见：`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_summary.md`

</details>

offline 吞吐这轮没有变得更“魔法”，依然是：比 vLLM 快不少，但落后 SGLang / TRT-LLM 一截：

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/offline_throughput_compare.png)

<details>
<summary>offline 原始数据（吞吐表）</summary>

见：`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/offline_summary.md`

</details>

---

## 这 3ms 差距我怎么理解

TTFT 里混着三类东西：

1. 服务侧开销：tokenize、admit、排队、SSE 首包
2. prefill 计算：真正的 forward（这块其实最“硬”）
3. 调度等待：prefill 何时开始跑、prefill 完成后第一步 decode 何时被调度

这轮的 profile 看下来，我更倾向于把剩下的差距归到 **prefill 常数项** 上：TensorRT-LLM 的 prefill 走的是 TensorRT engine，kernel 组合更紧、更少 launch、更少中间张量；我们这边虽然也用了 flashinfer paged prefill meta/fast plan，把很多重复工作摁住了，但整体还是 PyTorch eager + 多个 kernel 组合，常数项很难完全抹平。

所以这篇的目标不是“写个奇技淫巧就一定追上 TRT-LLM”，而是把 TTFT 的尾巴尽量往下压，把那些明显会制造抖动的点扫干净。

---

## 一条能稳定带来收益的路径

这一轮把 TTFT P99 压到 11ms 出头，本质上是一条很工程、但很有效的链路：

1. **prefill chunk 拉大**：把 `prefill_chunk_size` 拉到 1024（在这条 trace 里 prompt 基本都能塞进一个 chunk，TTFT 直接少掉“多轮 prefill chunk”的等待）
2. **decode batch 控制**：online 场景用 `max_batch_size=16`，在不把 TPOT 拖太多的前提下把吞吐稳住
3. **CUDA graph 预热**：提前 capture batch=1..16 的 decode graph，避免首波请求撞到 capture 抖动
4. **prefill 优先级**：`prefill_priority_threshold=1`，prefill queue 一旦有活就别让 decode 抢跑
5. **overlap 的 prefill completion 优先**：prefill 完成的 chunk 尽快“落地”，别让 completion 在 pending 里拖着
6. **mp idle keepalive**：这个不是提速大头，但能把某些边缘抖动压掉一点

过程图我不再重复画一堆，直接看这四个阶段的曲线会更直观：

- warmup cg16：`/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/online_latency_p99_only.png`
- pprio1：`/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/online_latency_p99_only.png`
- eager prefill / pfront：`/assets/images/posts/2026-01-01-ttft-p99-077/pfront/online_latency_p99_only.png`
- 最终（含 idle keepalive）：`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_p99_only.png`

---

## 失败尝试 1：+query decode

我一开始的直觉很朴素：overlap 模式下，我们处理 completion 的时候有 `synchronize()`，如果把 decode completion 改成 “event ready 才处理”，是不是能减少 CPU block，从而让 TTFT 更稳？

结果是反过来的：P99 直接炸掉。

![](/assets/images/posts/2026-01-01-ttft-p99-077/qdec/online_latency_p99_only.png)

`scale=0.4` 的数据很离谱：

| backend | TTFT p90 | TTFT p99 | ITL p99 |
|---|---:|---:|---:|
| roseinfer | 10.33 | 11.53 | 2.38 |
| roseinfer (+query decode) | 52.02 | 74.17 | 44.92 |

这条优化我最终把它定性成：**不仅负优化，而且有稳定性风险**。原因大概是：

- decode completion 不落地，会让 pending/future map 的积压变成一种“隐形队列”
- 极端情况下会出现乱序放大，甚至把一些错误异步延后到后面才报出来

profile 文件我也留着（方便以后对照）：

- `outputs/benchmarks/serving/online_20260103_062023/profile_manifest.json`（torch）
- `outputs/benchmarks/serving/online_20260103_061617/profiles/nsys/`（nsys，曾经触发过一次 device-side assert）

---

## 失败尝试 2：+single-chunk fp

这条的动机也很直觉：既然 prompt 大概率是一整个 chunk，那是不是可以走 eager prefill + register KV 的路径（绕开 flashinfer paged prefill）？

它确实把 TTFT 的 p90 压了一点，但 **p99 被一个极端 outlier 拉爆**，同时 offline 吞吐也明显掉。

![](/assets/images/posts/2026-01-01-ttft-p99-077/scfp/online_latency_p99_only.png)

`scale=0.4`（只看 TTFT）：

| backend | TTFT p90 | TTFT p99 |
|---|---:|---:|
| roseinfer | 10.33 | 11.53 |
| roseinfer (+single-chunk fp) | 9.93 | 13.52 |

offline 也掉得很明显：

![](/assets/images/posts/2026-01-01-ttft-p99-077/scfp/offline_throughput_compare.png)

这条的结论很简单：**看起来像捷径，但对 P99 不友好**。我更愿意把它当作“研究 KV register 这条路的参考”，而不是一个能直接开在默认配置里的优化。

对应 profile：

- `outputs/benchmarks/serving/online_20260103_071801/profile_manifest.json`
- `outputs/benchmarks/serving/offline_20260103_071947/profile_manifest.json`

---

## 失败尝试 3：prefill bsz8

我还试了一个更像“调参”的东西：把 prefill 最大 batch 从 16 限到 8，希望减少 prefill 单次 kernel 的时间，换 TTFT。

结论也是负的：p90 没什么优势，但 p99 反而更差。

![](/assets/images/posts/2026-01-01-ttft-p99-077/pmb8/online_latency_p99_only.png)

`scale=0.4`：

| backend | TTFT p90 | TTFT p99 |
|---|---:|---:|
| roseinfer | 10.33 | 11.53 |
| roseinfer (+prefill bsz8) | 10.11 | 13.37 |

对应 profile：

- `outputs/benchmarks/serving/online_20260103_075303/profile_manifest.json`

---

## 尝试 4：+shared block tables

这一条是我觉得“很像正经优化”的：**paged-KV 的 block table 本质上是 per-seq 的映射，不是 per-layer 的**。

但我们现在的实现里，为了图省事，global block table 是按 layer 存的（形态更接近 `[L, S, B]`），每次 sync dirty rows 都要在 CPU 侧 loop 一遍 layer，再 `index_copy_` 一遍 GPU。直觉上这块应该可以压掉不少常数。

做法很简单：把 global block table 改成 `[S, B]`，只 sync 一次；forward 需要的 `block_tables` 列表就用同一个 tensor 重复引用 `num_layers` 次（`[global_bt] * L`）。

开关名：`--shared-block-tables`（默认关闭，方便 A/B）。

先看数据（这里为了不把图塞爆，我只拿 `+batch16 +chunk1024` 这条线上做对比）：  

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_latency_compare.png)

只看 P99 / P90：

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_latency_p90_only.png)

`scale=0.4` 摘出来一行（TTFT/TPOT/ITL/E2E 都看 p90/p99）：

| backend | TTFT p90 | TTFT p99 | TPOT p90 | TPOT p99 | E2E p90 | E2E p99 |
|---|---:|---:|---:|---:|---:|---:|
| roseinfer (+batch16, +chunk1024) | 16.07 | 25.90 | 1.52 | 1.87 | 195.33 | 245.96 |
| roseinfer (+batch16, +chunk1024, +shared BT) | 15.75 | 26.54 | 1.51 | 1.98 | 196.37 | 259.07 |

结论比较“反直觉”：**p90 稍微好一点，但 p99 没压下去，甚至有点回弹**。这意味着：

- 我猜的那部分 per-layer block table sync 常数，确实存在，但它并不是我们 TTFT 长尾的决定因素；
- 或者说，减少 sync 次数并不等价于减少尾巴（尾巴更像是 scheduler/graph capture/其他 host-side 抖动在放大）。

offline 吞吐这条倒是有一点点小涨（更像噪声级别），放这儿做个记录：

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/offline_throughput_compare.png)

profile 我也留着（方便后面对照）：

- `outputs/benchmarks/serving/online_20260103_225834/profile_manifest.json`
- `outputs/benchmarks/serving/offline_20260103_230817/profile_manifest.json`

---

## profile 文件怎么打开

我自己常用两条：

- torch profile：直接看 `profile_manifest.json` 里列的 `trace.json` / `*.trace.json.gz`，用 Chrome tracing 或者 Perfetto 打开
- nsys：`nsys-ui` 打开 `nsys.nsys-rep`，或者用命令行先过一遍摘要：`nsys stats --report nvtx_sum nsys.nsys-rep`

这轮新增的 profile 都在上面各个小节里贴了路径，直接点进去就能找到。

---

## 小结

这一轮最有价值的收获其实不是某一个开关，而是一个很清晰的结论：

- 我们已经能把 TTFT 的长尾压到比 SGLang 更好，但 **要追平 TRT-LLM，剩下的主要是 prefill 的常数项差距**。
- scheduler 层面再抠几个开关，能带来的是 0.x ms 级别的波动；要再往下走，恐怕得在 prefill kernel 组合、KV 写入路径、甚至更激进的图编译上动刀。

下一篇我准备继续围绕 “prefill 常数项” 做更硬核一点的尝试：要么把 prefill 的 CPU/GPU 数据流压得更平，要么直接想办法减少 prefill kernel 数量。TTFT P99 这条线我会继续追下去。

---

## （2026-01-05 更新）全量优化特性回归：只分析已有 outputs（不跑新 benchmark）

这一节是为了把 “到底哪个特性提升最大、提升在哪个指标上、为什么会提升/为什么会没有提升” 这件事说清楚。

强调三遍：**下面所有数据都来自已有执行结果**，只做“读取 JSON → 统一口径算百分比 → 画图/排表”，不再启动任何 server、不再跑任何新的 benchmark。

### 数据来源（原始数据/图表都在 repo 里）

1. merged JSON（结构化原始结果）：
   - online：`outputs/benchmarks/serving/online_merged_077_*.json`
   - offline：`outputs/benchmarks/serving/offline_merged_077_*.json`
2. per-feature 图与 summary（当时 benchmark 脚本直接产出的“原始表格 + 对比图”）：
   - `outputs/benchmarks/serving/figures_077_<feature>/`
   - 为了博客引用方便，我把这些文件拷贝到了本文 assets 下同名目录（例：`/assets/images/posts/2026-01-01-ttft-p99-077/aflush/`、`/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/`）
3. 本文新增的“全量对比图 + 排名表”（由 merged JSON 直接生成，便于统一口径）：
   - online（scale=0.4）：`/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_deltas.csv`
   - offline（vs inproc）：`/assets/images/posts/2026-01-01-ttft-p99-077/analysis/offline_deltas.csv`
   - 全量排名（可直接点开看表）：`/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_rankings.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/analysis/offline_vs_inproc_rankings.md`

### 口径（读表之前先把话说死）

1. **online 的“优化百分比”**：对每个 feature，我都在该 feature 的 merged JSON 内部选一对最接近的 `base`/`variant`（基本只差这个开关），并按下面公式算：

$$
\mathrm{improve\%} = \frac{\mathrm{base}-\mathrm{variant}}{\mathrm{base}} \times 100\%
$$

所以：

- **improve% 为正**：延迟下降（更快）
- **improve% 为负**：延迟上升（更慢）

2. **注意：不同 feature 的 base 可能不是同一套配置**。原因很简单：这些结果来自我当时按“优化链路”逐步做 A/B 的过程，有些是在更早的基线跑的，有些是在更后的基线跑的。  
所以“严格排序”只能回答：“**在各自测试上下文里，这个开关相对 base 贡献了多少**”，而不是一个严格意义的“把所有开关都叠在同一个 base 上再排序”的公平大赛（那需要重新跑一轮统一矩阵）。

3. **重点关注 scale=0.4**：`scale` 越小负载越大（trace 被压缩得更“挤”），所以最能放大 tail。本文重点盯：

- `scale=0.4` 的 **TTFT / E2E**：P90、P99（你关心的重点）
- 同时把 ITL/TPOT 也按同口径排出来（完整表见 `analysis/online_scale0.4_rankings.md`）

4. **offline 的“优化百分比”**：吞吐（tok/s）越大越好。本文默认用你给的口径：相对 `roseinfer+inproc`：

$$
\mathrm{improve\%} = \frac{\mathrm{variant}-\mathrm{inproc}}{\mathrm{inproc}} \times 100\%
$$

---

## Online：scale=0.4 全量对比图（严格排序见下节表格）

先给全量对比图（每个条形代表一个 feature 的 improve%，绿色=变快，红色=变慢）。

### 你最关心的：TTFT / E2E（P90/P99）

TTFT P90：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_ttft_p90_improve.png)

TTFT P99：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_ttft_p99_improve.png)

E2E P90：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_e2e_p90_improve.png)

E2E P99：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_e2e_p99_improve.png)

### 作为补充：ITL / TPOT（P90）

ITL P90：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_itl_p90_improve.png)

TPOT P90：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_tpot_p90_improve.png)

### 一眼看 tradeoff：热力图 + TTFT vs E2E 散点

P90 热力图（横轴 4 个指标；纵轴 feature；颜色越绿越好）：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_heatmap_p90.png)

P99 热力图（重点看 tail）：  

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_heatmap_p99.png)

把 “TTFT 改善” 和 “E2E 改善” 放到同一张散点图上（右上角才是两者一起变好）：

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_scatter_ttft_e2e_p90.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_scatter_ttft_e2e_p99.png)

### 先说结论（scale=0.4 的直觉版）

1. **TTFT tail（P99）最大头基本都属于“减少首轮抖动/冷启动/host-side 常数”的开关**：例如 CUDA graph warmup（`wupcg16`）、更激进的 prefill/overlap 调度（`peager`）、prefill 优先级（`pprio1`）。
2. **E2E（尤其 P90）最大的改善来自“把 steady-state 的 decode+streaming 后半段做得更顺”**：例如 `aflush` 对 E2E P90 的提升非常大（因为它直接减少了 token 流水线背压/事件堆积）。
3. **“吞吐类优化”不一定对 TTFT tail 友好**：典型例子 `chunkbkt`——它能显著降低 ITL/TPOT/E2E（因为减少 padding、提高 batch 效率），但 TTFT P99 反而变差（更像是把某些请求在队列里“延后”，造成 head-of-line 的尾巴）。

---

## Online：scale=0.4 严格排序（TTFT/ITL/TPOT/E2E，P90/P99）

说明：下面表格里 base/variant 都是该 feature 的 merged JSON 内部最接近的一对；`improve` 为正表示更快。

> 全量表（包含 p50/p90/p99）也可以直接点开：  
> `/assets/images/posts/2026-01-01-ttft-p99-077/analysis/online_scale0.4_rankings.md`

<details>
<summary>TTFT P90（scale=0.4）严格排序</summary>

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | peager | +8.25% | 10.72 | 9.84 |
| 2 | pprio1 | +6.84% | 11.51 | 10.72 |
| 3 | pstream | +5.66% | 10.72 | 10.11 |
| 4 | wuppf16 | +5.46% | 10.72 | 10.14 |
| 5 | aflush | +4.81% | 10.72 | 10.20 |
| 6 | scfp | +3.89% | 10.33 | 9.93 |
| 7 | admit64 | +3.82% | 10.72 | 10.31 |
| 8 | rnopast | +2.60% | 10.72 | 10.44 |
| 9 | fbsync | +2.57% | 15.66 | 15.25 |
| 10 | wupcg16 | +2.55% | 11.81 | 11.51 |
| 11 | pmb8 | +2.13% | 10.33 | 10.11 |
| 12 | opool | +2.13% | 10.72 | 10.49 |
| 13 | tokbuf | +0.41% | 10.72 | 10.68 |
| 14 | tokbuf2 | -0.91% | 10.33 | 10.43 |
| 15 | chunkbkt | -1.13% | 11.51 | 11.64 |
| 16 | pfront | -1.21% | 9.84 | 9.96 |
| 17 | dcap8 | -1.32% | 11.51 | 11.66 |
| 18 | nopad | -1.85% | 11.51 | 11.72 |
| 19 | pcap8 | -2.41% | 9.84 | 10.07 |
| 20 | pbuild | -2.45% | 11.51 | 11.79 |
| 21 | pbuild2 | -2.64% | 10.33 | 10.60 |
| 22 | pscan4 | -3.18% | 9.84 | 10.15 |
| 23 | lastlogits_peager | -3.27% | 9.84 | 10.16 |
| 24 | pslots | -3.66% | 11.51 | 11.93 |
| 25 | lastlogits | -4.33% | 11.51 | 12.01 |
| 26 | idleka | -5.02% | 9.84 | 10.33 |
| 27 | fanopast | -6.43% | 11.51 | 12.25 |
| 28 | bffwd | -16.66% | 11.51 | 13.43 |
| 29 | qdec | -403.58% | 10.33 | 52.02 |
| 30 | evtq | -424.32% | 9.84 | 51.57 |

</details>

<details>
<summary>TTFT P99（scale=0.4）严格排序（最关注的 tail）</summary>

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | wupcg16 | +24.67% | 19.65 | 14.80 |
| 2 | peager | +13.19% | 13.40 | 11.64 |
| 3 | pprio1 | +9.44% | 14.80 | 13.40 |
| 4 | fbsync | +6.55% | 26.17 | 24.46 |
| 5 | pbuild | +1.82% | 14.80 | 14.53 |
| 6 | idleka | +0.90% | 11.64 | 11.53 |
| 7 | lastlogits | -1.04% | 14.80 | 14.95 |
| 8 | pstream | -1.56% | 13.40 | 13.61 |
| 9 | pfront | -3.03% | 11.64 | 11.99 |
| 10 | wuppf16 | -3.27% | 13.40 | 13.84 |
| 11 | opool | -4.51% | 13.40 | 14.01 |
| 12 | aflush | -4.55% | 13.40 | 14.01 |
| 13 | pscan4 | -6.23% | 11.64 | 12.36 |
| 14 | lastlogits_peager | -6.84% | 11.64 | 12.43 |
| 15 | admit64 | -6.99% | 13.40 | 14.34 |
| 16 | nopad | -10.91% | 14.80 | 16.42 |
| 17 | pcap8 | -12.20% | 11.64 | 13.06 |
| 18 | tokbuf | -13.32% | 13.40 | 15.19 |
| 19 | tokbuf2 | -14.84% | 11.53 | 13.24 |
| 20 | pmb8 | -15.93% | 11.53 | 13.37 |
| 21 | rnopast | -16.51% | 13.40 | 15.62 |
| 22 | dcap8 | -17.13% | 14.80 | 17.34 |
| 23 | scfp | -17.24% | 11.53 | 13.52 |
| 24 | chunkbkt | -17.56% | 14.80 | 17.40 |
| 25 | pslots | -18.19% | 14.80 | 17.49 |
| 26 | pbuild2 | -19.33% | 11.53 | 13.76 |
| 27 | fanopast | -26.43% | 14.80 | 18.71 |
| 28 | bffwd | -40.41% | 14.80 | 20.78 |
| 29 | evtq | -436.22% | 11.64 | 62.39 |
| 30 | qdec | -543.20% | 11.53 | 74.17 |

</details>

<details>
<summary>ITL P90 / P99（scale=0.4）严格排序</summary>

#### ITL P90

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | evtq | +23.60% | 1.52 | 1.16 |
| 2 | qdec | +13.90% | 1.52 | 1.31 |
| 3 | fbsync | +9.83% | 1.54 | 1.38 |
| 4 | chunkbkt | +9.21% | 1.52 | 1.38 |
| 5 | aflush | +8.90% | 1.52 | 1.39 |
| 6 | pbuild | +1.55% | 1.52 | 1.50 |
| 7 | pstream | +1.08% | 1.52 | 1.50 |
| 8 | pslots | +0.90% | 1.52 | 1.51 |
| 9 | dcap8 | +0.78% | 1.52 | 1.51 |
| 10 | admit64 | +0.57% | 1.52 | 1.51 |
| 11 | scfp | +0.56% | 1.52 | 1.51 |
| 12 | nopad | +0.56% | 1.52 | 1.51 |
| 13 | opool | +0.54% | 1.52 | 1.51 |
| 14 | fanopast | +0.48% | 1.52 | 1.52 |
| 15 | pbuild2 | +0.47% | 1.52 | 1.52 |
| 16 | rnopast | +0.36% | 1.52 | 1.52 |
| 17 | pfront | +0.26% | 1.52 | 1.51 |
| 18 | peager | +0.23% | 1.52 | 1.52 |
| 19 | lastlogits_peager | +0.19% | 1.52 | 1.51 |
| 20 | wuppf16 | +0.18% | 1.52 | 1.52 |
| 21 | pprio1 | +0.11% | 1.52 | 1.52 |
| 22 | pmb8 | +0.07% | 1.52 | 1.52 |
| 23 | tokbuf2 | +0.01% | 1.52 | 1.52 |
| 24 | pcap8 | -0.10% | 1.52 | 1.52 |
| 25 | tokbuf | -0.28% | 1.52 | 1.53 |
| 26 | bffwd | -0.29% | 1.52 | 1.53 |
| 27 | lastlogits | -0.36% | 1.52 | 1.53 |
| 28 | idleka | -0.38% | 1.52 | 1.52 |
| 29 | pscan4 | -0.38% | 1.52 | 1.52 |
| 30 | wupcg16 | -9.67% | 1.39 | 1.52 |

#### ITL P99

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | fbsync | +14.60% | 2.73 | 2.33 |
| 2 | chunkbkt | +7.74% | 2.55 | 2.36 |
| 3 | aflush | +6.11% | 2.53 | 2.38 |
| 4 | peager | +4.42% | 2.53 | 2.42 |
| 5 | pbuild | +3.27% | 2.55 | 2.47 |
| 6 | lastlogits_peager | +2.73% | 2.42 | 2.35 |
| 7 | pslots | +2.50% | 2.55 | 2.49 |
| 8 | lastlogits | +1.85% | 2.55 | 2.51 |
| 9 | dcap8 | +1.79% | 2.55 | 2.51 |
| 10 | idleka | +1.71% | 2.42 | 2.38 |
| 11 | pscan4 | +1.38% | 2.42 | 2.39 |
| 12 | pcap8 | +1.30% | 2.42 | 2.39 |
| 13 | scfp | +1.23% | 2.38 | 2.35 |
| 14 | pprio1 | +0.83% | 2.55 | 2.53 |
| 15 | pfront | +0.52% | 2.42 | 2.41 |
| 16 | pmb8 | +0.51% | 2.38 | 2.37 |
| 17 | pbuild2 | -0.01% | 2.38 | 2.38 |
| 18 | fanopast | -0.43% | 2.55 | 2.56 |
| 19 | opool | -0.56% | 2.53 | 2.55 |
| 20 | admit64 | -0.80% | 2.53 | 2.55 |
| 21 | wuppf16 | -1.95% | 2.53 | 2.58 |
| 22 | tokbuf | -1.97% | 2.53 | 2.58 |
| 23 | nopad | -2.17% | 2.55 | 2.61 |
| 24 | rnopast | -2.20% | 2.53 | 2.59 |
| 25 | bffwd | -3.28% | 2.55 | 2.64 |
| 26 | tokbuf2 | -5.11% | 2.38 | 2.50 |
| 27 | wupcg16 | -10.19% | 2.32 | 2.55 |
| 28 | pstream | -13.40% | 2.53 | 2.87 |
| 29 | evtq | -1628.16% | 2.42 | 41.83 |
| 30 | qdec | -1788.18% | 2.38 | 44.92 |

</details>

<details>
<summary>TPOT P90 / P99（scale=0.4）严格排序</summary>

#### TPOT P90

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | fbsync | +6.99% | 1.52 | 1.41 |
| 2 | aflush | +6.15% | 1.43 | 1.35 |
| 3 | chunkbkt | +6.12% | 1.44 | 1.35 |
| 4 | lastlogits_peager | +0.53% | 1.44 | 1.43 |
| 5 | pfront | +0.53% | 1.44 | 1.43 |
| 6 | scfp | +0.29% | 1.44 | 1.43 |
| 7 | idleka | +0.28% | 1.44 | 1.44 |
| 8 | pprio1 | +0.20% | 1.44 | 1.43 |
| 9 | nopad | +0.16% | 1.44 | 1.43 |
| 10 | pscan4 | +0.05% | 1.44 | 1.44 |
| 11 | tokbuf2 | +0.03% | 1.44 | 1.44 |
| 12 | pbuild2 | +0.01% | 1.44 | 1.44 |
| 13 | pcap8 | +0.01% | 1.44 | 1.44 |
| 14 | pmb8 | -0.02% | 1.44 | 1.44 |
| 15 | pbuild | -0.08% | 1.44 | 1.44 |
| 16 | rnopast | -0.26% | 1.43 | 1.44 |
| 17 | dcap8 | -0.31% | 1.44 | 1.44 |
| 18 | wuppf16 | -0.32% | 1.43 | 1.44 |
| 19 | admit64 | -0.37% | 1.43 | 1.44 |
| 20 | pstream | -0.38% | 1.43 | 1.44 |
| 21 | opool | -0.40% | 1.43 | 1.44 |
| 22 | tokbuf | -0.43% | 1.43 | 1.44 |
| 23 | lastlogits | -0.43% | 1.44 | 1.44 |
| 24 | peager | -0.56% | 1.43 | 1.44 |
| 25 | fanopast | -0.59% | 1.44 | 1.45 |
| 26 | pslots | -0.59% | 1.44 | 1.45 |
| 27 | bffwd | -1.91% | 1.44 | 1.46 |
| 28 | wupcg16 | -3.66% | 1.39 | 1.44 |
| 29 | evtq | -13.98% | 1.44 | 1.64 |
| 30 | qdec | -27.84% | 1.44 | 1.84 |

#### TPOT P99

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | fbsync | +10.98% | 1.99 | 1.77 |
| 2 | wupcg16 | +6.17% | 1.69 | 1.58 |
| 3 | aflush | +5.71% | 1.57 | 1.48 |
| 4 | chunkbkt | +3.70% | 1.58 | 1.52 |
| 5 | lastlogits_peager | +1.37% | 1.58 | 1.55 |
| 6 | idleka | +1.29% | 1.58 | 1.56 |
| 7 | pbuild2 | +1.27% | 1.56 | 1.54 |
| 8 | pmb8 | +0.78% | 1.56 | 1.54 |
| 9 | pscan4 | +0.55% | 1.58 | 1.57 |
| 10 | pprio1 | +0.54% | 1.58 | 1.57 |
| 11 | scfp | +0.38% | 1.56 | 1.55 |
| 12 | pfront | +0.33% | 1.58 | 1.57 |
| 13 | opool | +0.23% | 1.57 | 1.57 |
| 14 | pbuild | +0.11% | 1.58 | 1.58 |
| 15 | peager | -0.12% | 1.57 | 1.58 |
| 16 | lastlogits | -0.32% | 1.58 | 1.59 |
| 17 | wuppf16 | -0.34% | 1.57 | 1.58 |
| 18 | pstream | -0.41% | 1.57 | 1.58 |
| 19 | nopad | -1.42% | 1.58 | 1.60 |
| 20 | pcap8 | -1.45% | 1.58 | 1.60 |
| 21 | pslots | -1.68% | 1.58 | 1.61 |
| 22 | admit64 | -1.91% | 1.57 | 1.60 |
| 23 | dcap8 | -2.20% | 1.58 | 1.62 |
| 24 | rnopast | -2.30% | 1.57 | 1.61 |
| 25 | tokbuf | -2.45% | 1.57 | 1.61 |
| 26 | fanopast | -3.36% | 1.58 | 1.64 |
| 27 | tokbuf2 | -3.56% | 1.56 | 1.61 |
| 28 | bffwd | -6.40% | 1.58 | 1.68 |
| 29 | qdec | -32.80% | 1.56 | 2.07 |
| 30 | evtq | -50.12% | 1.58 | 2.37 |

</details>

<details>
<summary>E2E P90 / P99（scale=0.4）严格排序（你关心的另一个重点）</summary>

#### E2E P90

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | aflush | +9.63% | 186.40 | 168.45 |
| 2 | chunkbkt | +9.06% | 186.81 | 169.88 |
| 3 | fbsync | +6.74% | 197.33 | 184.03 |
| 4 | scfp | +0.64% | 186.91 | 185.72 |
| 5 | wuppf16 | +0.28% | 186.40 | 185.88 |
| 6 | nopad | +0.26% | 186.81 | 186.32 |
| 7 | lastlogits_peager | +0.26% | 186.19 | 185.71 |
| 8 | pprio1 | +0.22% | 186.81 | 186.40 |
| 9 | rnopast | +0.20% | 186.40 | 186.02 |
| 10 | pbuild2 | +0.17% | 186.91 | 186.60 |
| 11 | pstream | +0.13% | 186.40 | 186.16 |
| 12 | peager | +0.11% | 186.40 | 186.19 |
| 13 | pbuild | +0.11% | 186.81 | 186.60 |
| 14 | tokbuf2 | +0.02% | 186.91 | 186.86 |
| 15 | pslots | -0.00% | 186.81 | 186.81 |
| 16 | pfront | -0.03% | 186.19 | 186.24 |
| 17 | tokbuf | -0.07% | 186.40 | 186.53 |
| 18 | admit64 | -0.09% | 186.40 | 186.56 |
| 19 | pmb8 | -0.13% | 186.91 | 187.15 |
| 20 | opool | -0.26% | 186.40 | 186.89 |
| 21 | dcap8 | -0.27% | 186.81 | 187.31 |
| 22 | lastlogits | -0.38% | 186.81 | 187.52 |
| 23 | idleka | -0.39% | 186.19 | 186.91 |
| 24 | pscan4 | -0.51% | 186.19 | 187.13 |
| 25 | pcap8 | -0.51% | 186.19 | 187.15 |
| 26 | fanopast | -0.61% | 186.81 | 187.95 |
| 27 | bffwd | -2.14% | 186.81 | 190.80 |
| 28 | wupcg16 | -3.18% | 181.05 | 186.81 |
| 29 | evtq | -22.50% | 186.19 | 228.08 |
| 30 | qdec | -32.51% | 186.91 | 247.67 |

#### E2E P99

| rank | feature | improve | base | variant |
|---:|---|---:|---:|---:|
| 1 | fbsync | +8.21% | 259.45 | 238.14 |
| 2 | wupcg16 | +7.77% | 224.63 | 207.18 |
| 3 | aflush | +4.49% | 206.48 | 197.21 |
| 4 | chunkbkt | +1.55% | 207.18 | 203.97 |
| 5 | peager | +0.78% | 206.48 | 204.87 |
| 6 | pbuild2 | +0.70% | 205.67 | 204.24 |
| 7 | admit64 | +0.50% | 206.48 | 205.45 |
| 8 | opool | +0.46% | 206.48 | 205.52 |
| 9 | pprio1 | +0.34% | 207.18 | 206.48 |
| 10 | pmb8 | +0.34% | 205.67 | 204.98 |
| 11 | pstream | +0.26% | 206.48 | 205.95 |
| 12 | wuppf16 | +0.19% | 206.48 | 206.08 |
| 13 | tokbuf2 | +0.07% | 205.67 | 205.54 |
| 14 | pscan4 | -0.01% | 204.87 | 204.88 |
| 15 | tokbuf | -0.02% | 206.48 | 206.51 |
| 16 | pfront | -0.15% | 204.87 | 205.18 |
| 17 | rnopast | -0.24% | 206.48 | 206.98 |
| 18 | lastlogits_peager | -0.31% | 204.87 | 205.50 |
| 19 | idleka | -0.39% | 204.87 | 205.67 |
| 20 | pbuild | -0.52% | 207.18 | 208.26 |
| 21 | lastlogits | -1.23% | 207.18 | 209.74 |
| 22 | nopad | -1.87% | 207.18 | 211.05 |
| 23 | scfp | -2.73% | 205.67 | 211.29 |
| 24 | pslots | -2.78% | 207.18 | 212.95 |
| 25 | dcap8 | -3.03% | 207.18 | 213.46 |
| 26 | pcap8 | -3.70% | 204.87 | 212.46 |
| 27 | fanopast | -4.10% | 207.18 | 215.67 |
| 28 | bffwd | -6.91% | 207.18 | 221.49 |
| 29 | qdec | -46.52% | 205.67 | 301.34 |
| 30 | evtq | -47.91% | 204.87 | 303.03 |

</details>

---

## Offline：吞吐（total tok/s）严格排序（vs `roseinfer+inproc`）

offline 的特点是：没有真实在线队列、没有 SSE 首包、也没有“trace 被压缩导致的 burst 抖动”，所以很多 “压尾巴/减抖动” 的开关在 offline 上看不到收益；相反，**减少 forward 次数、减少 kernel/拷贝/张量构造** 的优化更容易在 offline 体现为 tok/s 提升。

总览（绿色=吞吐更高）：  

![](/assets/images/posts/2026-01-01-ttft-p99-077/analysis/offline_total_tokps_vs_inproc_improve.png)

严格排序表：

| rank | feature | improve | inproc tok/s | variant tok/s |
|---:|---|---:|---:|---:|
| 1 | lastlogits | +10.53% | 65314.13 | 72191.59 |
| 2 | fanopast | +3.26% | 65314.13 | 67445.26 |
| 3 | nopad | +2.99% | 65314.13 | 67269.86 |
| 4 | rnopast | +1.52% | 65314.13 | 66308.00 |
| 5 | tokbuf | +0.66% | 65314.13 | 65748.01 |
| 6 | pstream | +0.30% | 65314.13 | 65511.07 |
| 7 | qdec | -0.04% | 65314.13 | 65287.94 |
| 8 | pprio1 | -0.20% | 65314.13 | 65183.33 |
| 9 | chunkbkt | -0.75% | 65314.13 | 64822.70 |
| 10 | wupcg16 | -0.80% | 65314.13 | 64789.61 |
| 11 | bffwd | -1.02% | 65314.13 | 64646.62 |
| 12 | pbuild | -1.13% | 65314.13 | 64577.50 |
| 13 | pslots | -1.25% | 65314.13 | 64496.48 |
| 14 | idleka | -1.26% | 65314.13 | 64489.86 |
| 15 | pfront | -1.38% | 65314.13 | 64415.37 |
| 16 | aflush | -1.50% | 65314.13 | 64334.99 |
| 17 | peager | -1.51% | 65314.13 | 64328.54 |
| 18 | opool | -1.62% | 65314.13 | 64254.11 |
| 19 | pbuild2 | -1.65% | 65314.13 | 64238.98 |
| 20 | tokbuf2 | -1.86% | 65314.13 | 64101.55 |
| 21 | fbsync | -2.12% | 65314.13 | 63931.33 |
| 22 | scfp | -9.31% | 65314.13 | 59230.57 |
| 23 | pcap8 | -27.82% | 65314.13 | 47143.17 |

> 更完整的 offline 原始表（包含 prefill_s/decode_s/total_s）见：  
> `/assets/images/posts/2026-01-01-ttft-p99-077/analysis/offline_deltas.csv`

---

## 每个特性到底在干什么？为什么会提升/为什么会没提升？（零基础版）

这一节我按“影响链路”把 feature 分组讲清楚。先把 online 的一条请求拆开看（非常重要）：

1. **入队阶段（CPU）**：HTTP/SSE 连接、tokenize、admit 进 scheduler
2. **prefill 阶段（GPU）**：把 prompt 的 token 喂进模型一次（或多次 chunk），写 KV
3. **first token**：采样/解码出第一个 token，并把它通过 SSE 发出去（这一步决定 TTFT）
4. **decode 阶段（GPU+CPU）**：循环生成后续 token，并持续 streaming（这决定 ITL/TPOT/E2E 的大头）

所以经验上：

- **TTFT** 主要受 1+2+3 影响（尤其是 2/3 的“冷启动/抖动”会被放大到 P99）。
- **ITL/TPOT** 主要受 4 影响（steady-state decode 的吞吐/调度/背压）。
- **E2E** 是 “TTFT + decode 全部 token” 的叠加：在 `scale=0.4` 这种重载下，任何 queueing/backpressure 都会直接反映到 E2E。

下面逐个 feature 解释，并贴出 `scale=0.4` 的关键数字（P90/P99）。如果你想看全 scales/p50/p90/p99 的原始表，直接点各目录的 `online_summary.md`/`offline_summary.md`。

### 1) chunked prefill 调度/公平性相关（最容易出现“平均变好但尾巴变坏”）

<details>
<summary><code>chunkbkt</code>：chunk 长度分桶调度（<code>--prefill-bucket-schedule</code>）</summary>

**做什么**：在 chunked prefill 的队列里，不再简单 FIFO 取前 `max_batch` 个，而是：

- 先从队列头“偷看”一段窗口（`--prefill-bucket-lookahead`）
- 计算每个 request 当前要 prefill 的 chunk 长度
- 以 2 的幂（并被 chunk_size 截断）做 bucket：`bucket_len = min(chunk_size, 2^ceil(log2(len)))`
- 优先凑齐一个“长度更一致”的 batch（减少 padding），没选中的 request 重新塞回队列尾部

实现点：`rosellm/roseinfer/engine.py` 的 `_select_prefill_batch()`（可以直接搜 `prefill_bucket_schedule`）。

**为什么会快**：prefill 的 attention/MLP 很吃 batch 内的 shape，一旦长度差太大就会有 padding 浪费（更长的序列决定 kernel 的有效工作量）。分桶调度本质上是在用“牺牲一点 FIFO 公平性”换“更一致的 prefill shape”，所以：

- **ITL/TPOT/E2E（尤其 P90）** 往往变好（吞吐更稳）
- **TTFT（尤其 P99）** 可能变差（某些短/长请求被反复 defer，形成 head-of-line 的尾巴）

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 11.64 | -1.13% | 14.80 | 17.40 | -17.56% |
| ITL (ms) | 1.52 | 1.38 | +9.21% | 2.55 | 2.36 | +7.74% |
| TPOT (ms) | 1.44 | 1.35 | +6.12% | 1.58 | 1.52 | +3.70% |
| E2E (ms) | 186.81 | 169.88 | +9.06% | 207.18 | 203.97 | +1.55% |

**怎么解读**：这就是典型的“吞吐类优化”：它确实让 decode 更顺（ITL/TPOT 降了很多），E2E P90 也显著下降；但 TTFT P99 变坏很明显——说明 `scale=0.4` 的 burst 下，队列里某些请求在 bucket 策略里更容易被“放到后面”，尾巴被放大了。

**offline（vs inproc）**：`total tok/s -0.75%`（65314.13 → 64822.70）。这也符合直觉：offline 没有真实在线队列，bucket 调度的收益变小；相反，多做一次 “lookahead + 分桶” 的调度逻辑可能只剩下额外开销。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/offline_summary.md`

</details>

<details>
<summary><code>pprio1</code>：prefill 优先级阈值（<code>--prefill-priority-threshold 1</code>）</summary>

**做什么**：当“待 prefill 的 chunk 数量 ≥ 阈值”时，调度顺序变成 **prefill 先跑、decode 后跑**（避免 decode 抢占 GPU 导致 prefill 排队）。

实现点：`rosellm/roseinfer/engine.py` 的 `prefill_first = ...` 分支（`prefill_priority_threshold`）。

**为什么会快**：TTFT 的核心风险是“prefill 被 decode 挤出去”，尤其在 `scale=0.4` 时 decode backlog 很容易把 prefill 的开始时间往后推。prefill 优先级本质上是在重载时把系统从“吞吐最大化”切换到“首包优先”，所以：

- TTFT（P90/P99）通常会更好
- ITL/TPOT 可能略受影响（因为 decode 被更频繁地打断）

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 10.72 | +6.84% | 14.80 | 13.40 | +9.44% |
| ITL (ms) | 1.52 | 1.52 | +0.11% | 2.55 | 2.53 | +0.83% |
| TPOT (ms) | 1.44 | 1.43 | +0.20% | 1.58 | 1.57 | +0.54% |
| E2E (ms) | 186.81 | 186.40 | +0.22% | 207.18 | 206.48 | +0.34% |

**怎么解读**：这是“尾巴友好”的调度开关，TTFT P99 直接明显下降；对 decode 指标影响很小（这条 trace 下几乎是纯收益）。

**offline（vs inproc）**：`total tok/s -0.20%`（65314.13 → 65183.33）。offline 里它几乎不可能带来收益：offline 没有“prefill vs decode 抢占”这种在线排队形态，优先级策略只剩下额外控制路径。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/offline_summary.md`

</details>

<details>
<summary><code>dcap8</code>：prefill 活跃时 cap decode batch（<code>--decode-batch-cap-on-prefill 8</code>）</summary>

**做什么**：只要系统里还有 prefill request，就把 decode 的 batch size 上限从 `max_batch_size` 临时降到 8（避免 decode 一次拿走太多 GPU 时间片）。

实现点：`rosellm/roseinfer/engine.py` 的 `_select_decode_batch()`（`decode_batch_cap_on_prefill`）。

**为什么可能会快**：直觉上它是“更激进的 prefill 优先”：decode batch 小一些，prefill 更容易插队成功，TTFT 应该更稳。

**为什么可能会慢**：decode batch cap 本质上是降吞吐；如果系统整体变慢，队列长度增加，反而会把 TTFT/E2E 的尾巴拖长（尤其 P99）。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 11.66 | -1.32% | 14.80 | 17.34 | -17.13% |
| ITL (ms) | 1.52 | 1.51 | +0.78% | 2.55 | 2.51 | +1.79% |
| TPOT (ms) | 1.44 | 1.44 | -0.31% | 1.58 | 1.62 | -2.20% |
| E2E (ms) | 186.81 | 187.31 | -0.27% | 207.18 | 213.46 | -3.03% |

**怎么解读**：在这套 trace 下，`dcap8` 更像是“吞吐不够导致整体排队变长”，tail 直接变差，属于不建议默认打开的开关。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/dcap8/online_summary.md`

</details>

---

### 2) prefill 输入构造/元数据开销（更像“抠常数项”，通常对 P90 更敏感）

<details>
<summary><code>pbuild</code>：prefill 输入 tensor 快速构造（<code>--prefill-fast-input-build</code>）</summary>

**做什么**：chunked prefill 需要构造 `input_ids/attention_mask/position_ids`。默认实现大量 Python list + `torch.tensor(list)`；`pbuild` 改成尽量用矢量化张量操作一次性生成（减少 Python 循环/分配）。

实现点：`rosellm/roseinfer/engine.py` 的 `prefill_chunk_sessions()`（可以搜 `prefill_fast_input_build`）。

**为什么会快**：这类优化主要省的是 CPU 侧准备开销，通常对 `scale=0.4` 这种“每秒很多请求、host 压力大”的场景更友好。但它省的不是 GPU 主计算，所以不会神奇地改变 TTFT tail。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 11.79 | -2.45% | 14.80 | 14.53 | +1.82% |
| ITL (ms) | 1.52 | 1.50 | +1.55% | 2.55 | 2.47 | +3.27% |
| TPOT (ms) | 1.44 | 1.44 | -0.08% | 1.58 | 1.58 | +0.11% |
| E2E (ms) | 186.81 | 186.60 | +0.11% | 207.18 | 208.26 | -0.52% |

**怎么解读**：ITL/TPOT 有小幅收益（更像“抠 CPU 常数”），但 TTFT P90 反而变差、P99 略好，整体接近噪声级别：说明在这条链路里，“构造张量”不是决定性瓶颈，收益容易被调度抖动淹没。

**offline（vs inproc）**：`total tok/s -1.13%`（65314.13 → 64577.50）。offline 更偏“纯算力吞吐”，这类 host-side 张量构造优化并不一定能带来正收益（反而可能因为额外分支/检查引入一点点开销）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/online_summary.md`

</details>

<details>
<summary><code>nopad</code>：避免扫描 attention_mask 非零位（<code>--flashinfer-paged-prefill-nopad-mask</code>）</summary>

**做什么**：flashinfer paged prefill 需要把“有效 token”的索引喂给 kernel。默认实现通过 `attention_mask.bool().reshape(-1).nonzero()` 扫一遍；`nopad` 改成用 flashinfer 提供的 `get_batch_indices_positions(qo_indptr, lengths, total_len)` 直接生成索引（避免一次全量 nonzero 扫描）。

实现点：`rosellm/roseinfer/engine.py` 的 `meta_nopad_mask` 分支（可以搜 `flashinfer_paged_prefill_nopad_mask`）。

**为什么会快**：`nonzero()` 本质上是一次全量扫描 + 索引输出，对 host/device 都可能产生额外开销；在 chunked prefill 频繁调用时，这类开销会积累成 TTFT/E2E 的常数项。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 11.72 | -1.85% | 14.80 | 16.42 | -10.91% |
| ITL (ms) | 1.52 | 1.51 | +0.56% | 2.55 | 2.61 | -2.17% |
| TPOT (ms) | 1.44 | 1.43 | +0.16% | 1.58 | 1.60 | -1.42% |
| E2E (ms) | 186.81 | 186.32 | +0.26% | 207.18 | 211.05 | -1.87% |

**怎么解读**：这条结果有点“反直觉”：按理说省掉 nonzero 扫描应该更稳，但在这个上下文里反而 TTFT tail 更差。比较合理的解释是：

- `nopad` 省的是一个相对固定的常数；但 `scale=0.4` 下 TTFT tail 更可能被调度/排队主导；
- 另外也可能存在实现细节：比如额外的 meta 张量构造/同步把收益抵掉，甚至引入新的波动。

因此它更像是“在某些场景有效的常数优化”，但不保证对 tail 一定友好。

**offline（vs inproc）**：`total tok/s +2.99%`（65314.13 → 67269.86）。offline 里 “省掉 nonzero 扫描/元数据构造” 的收益更容易体现为吞吐提升。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/nopad/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/nopad/offline_summary.md`

</details>

<details>
<summary><code>bffwd</code>：flashinfer begin-forward fastpath（<code>--flashinfer-paged-prefill-begin-forward</code>）</summary>

**做什么**：把 flashinfer paged prefill 的某些准备步骤前移/缓存（更像是“让 kernel 计划/元数据更早准备好”）。

**为什么可能会快**：减少每个 prefill batch 的 Python/plan 开销（常数项）。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 13.43 | -16.66% | 14.80 | 20.78 | -40.41% |
| ITL (ms) | 1.52 | 1.53 | -0.29% | 2.55 | 2.64 | -3.28% |
| TPOT (ms) | 1.44 | 1.46 | -1.91% | 1.58 | 1.68 | -6.40% |
| E2E (ms) | 186.81 | 190.80 | -2.14% | 207.18 | 221.49 | -6.91% |

**怎么解读**：这条在这份数据里是明显负优化，说明 begin-forward 这条路径要么不适配当前 workload，要么引入了额外同步/开销。实际工程上这类 “fastpath” 很容易出现“本来想省一次准备，结果因为缓存 miss/shape 变化反而更慢”的情况。

**offline（vs inproc）**：`total tok/s -1.02%`（65314.13 → 64646.62）。同样说明这条 fastpath 在当时的实现里更像是额外负担。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/offline_summary.md`

</details>

---

### 3) “no-past prefill” 类：专门优化“没有 past KV 的 prefill”

<details>
<summary><code>rnopast</code>：ragged nopast fastpath（<code>--flashinfer-paged-prefill-ragged-nopast</code>）</summary>

**做什么**：当 `start_lens` 全为 0（所有序列都没有 past KV）时，走更适合“纯 prefill”的 ragged 路径，避免按“带 past”的通用路径准备数据。

实现点：`rosellm/roseinfer/engine.py` 会计算 `meta_all_no_past` 并把开关传进 `PagedKVCache(...)`。

**为什么可能会快**：no-past 情况下可以简化 KV 索引/indptr 的形态，少做一些 gather/scatter 的准备工作。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.44 | +2.60% | 13.40 | 15.62 | -16.51% |
| ITL (ms) | 1.52 | 1.52 | +0.36% | 2.53 | 2.59 | -2.20% |
| TPOT (ms) | 1.43 | 1.44 | -0.26% | 1.57 | 1.61 | -2.30% |
| E2E (ms) | 186.40 | 186.02 | +0.20% | 206.48 | 206.98 | -0.24% |

**怎么解读**：P90 的 TTFT 有改善，但 P99 明显变差——这说明它省的是“平均常数”，但 tail 更可能来自调度/队列或极端 shape；同时 nopast fastpath 也可能在某些 corner case 引入更慢的路径（例如 plan/cache 形态不稳定）。

**offline（vs inproc）**：`total tok/s +1.52%`（65314.13 → 66308.00）。这更像是 “纯 prefill 路径更省常数项” 的直接体现。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/offline_summary.md`

</details>

<details>
<summary><code>fanopast</code>：flash-attn nopast fastpath（<code>--flashinfer-paged-prefill-flashattn-nopast</code>）</summary>

**做什么**：同样针对 no-past 的 prefill，但用 flash-attn 的实现路径。

**为什么可能会快**：在 no-past 这种最常见的“首轮 prefill”里，flash-attn 的 kernel 组合可能更高效，减少常数项/launch 数。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 12.25 | -6.43% | 14.80 | 18.71 | -26.43% |
| ITL (ms) | 1.52 | 1.52 | +0.48% | 2.55 | 2.56 | -0.43% |
| TPOT (ms) | 1.44 | 1.45 | -0.59% | 1.58 | 1.64 | -3.36% |
| E2E (ms) | 186.81 | 187.95 | -0.61% | 207.18 | 215.67 | -4.10% |

**怎么解读**：这份数据里它对 online 是明显负的（尤其 TTFT tail）。但有意思的是它对 offline 吞吐是正的（下面会提到）。这说明：**online 的 TTFT tail 不是“prefill kernel 本身够不够快”这么简单**，调度/排队/首包链路往往更关键。

**offline（vs inproc）**：`total tok/s +3.26%`（65314.13 → 67445.26）。这很符合 “纯吞吐” 的直觉：no-past 场景下的 kernel 组合更省常数项。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/offline_summary.md`

</details>

---

### 4) warmup 类：专门解决“冷启动抖动”（对 P99 影响巨大）

<details>
<summary><code>wupcg16</code>：decode CUDA graph 预热（<code>--warmup-cuda-graphs-max-batch 16</code>）</summary>

**做什么**：提前 capture batch=1..16 的 decode CUDA graphs，避免线上第一波请求触发 graph capture（那会有很明显的 tail）。

实现点：`rosellm/roseinfer/engine.py` 的 `warmup_paged_attention_cuda_graphs()`。

**为什么会快**：CUDA graph capture 本质上是一次“把一段 kernel launch 序列录下来”，首次发生时会带来额外同步/分配/编译开销；在重载 trace 下，这种开销会直接砸到 TTFT P99。

**实际结果（scale=0.4）**（注意：它的 base/variant 在更早的基线链路上，所以改善幅度会非常大）：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.81 | 11.51 | +2.55% | 19.65 | 14.80 | +24.67% |
| ITL (ms) | 1.39 | 1.52 | -9.67% | 2.32 | 2.55 | -10.19% |
| TPOT (ms) | 1.39 | 1.44 | -3.66% | 1.69 | 1.58 | +6.17% |
| E2E (ms) | 181.05 | 186.81 | -3.18% | 224.63 | 207.18 | +7.77% |

**怎么解读**：TTFT P99 的提升非常显著，这符合 “warmup 消掉冷启动抖动” 的直觉。其它指标的波动则取决于 capture 以后图的适配程度（某些 shape/路径可能没被覆盖，或者 capture 带来额外限制）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/offline_summary.md`

</details>

<details>
<summary><code>wuppf16</code>：prefill kernel 预热（<code>--warmup-prefill-batch-size 16</code> + <code>--warmup-prefill-lens ...</code>）</summary>

**做什么**：在 server 启动时，用一组典型的 prefill 长度（lens 列表）做一次 chunked prefill，提前触发 kernel 的编译/缓存路径，减少线上第一次遇到某个 shape 时的抖动。

实现点：`rosellm/roseinfer/engine.py` 的 `warmup_chunked_prefill()`（要求 CUDA + AMP + paged attention）。

**为什么可能会快**：和 CUDA graph warmup 类似，它主要是减少“第一次遇到某个 shape 的冷启动成本”，理论上更利好 TTFT tail。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.14 | +5.46% | 13.40 | 13.84 | -3.27% |
| ITL (ms) | 1.52 | 1.52 | +0.18% | 2.53 | 2.58 | -1.95% |
| TPOT (ms) | 1.43 | 1.44 | -0.32% | 1.57 | 1.58 | -0.34% |
| E2E (ms) | 186.40 | 185.88 | +0.28% | 206.48 | 206.08 | +0.19% |

**怎么解读**：它把 TTFT P90 压下去很明显，但 P99 反而略回弹——说明 prefill warmup 解决的是“常见 shape 的启动成本”，但 P99 更可能来自更极端的 shape/队列/调度；或者 warmup 覆盖不到真正导致 tail 的那部分路径。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/wuppf16/online_summary.md`

</details>

---

### 5) MP/事件/背压相关：经常直接影响 E2E（尤其 P90）

<details>
<summary><code>aflush</code>：flush admit token events（<code>--mp-flush-admit</code>）</summary>

**做什么**：在 mp engine loop 里，先把“刚 admit 的请求（以及它们已经产生的 token）”尽快通过事件队列 flush 给 server，再去跑下一轮 scheduler.step（也就是 GPU compute）。  
直觉上：**不要让 token 事件在 engine 侧堆积成一个隐形队列**。

实现点：`rosellm/roseinfer/mp.py` 中 `if flush_admit and batch: ... evt_q.put(...)` 的分支（在 `scheduler.step()` 之前 flush）。

**为什么会快**：这条优化主要不是省 GPU，而是减少：

- server 侧拿不到 token 时的等待（SSE/streaming 背压）
- engine/host 侧 token 事件积压导致的处理抖动（尤其大 burst 下）

所以它经常会同时改善 ITL/TPOT/E2E，而 TTFT 是否改善取决于“首 token 事件是否能更早落地”。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.20 | +4.81% | 13.40 | 14.01 | -4.55% |
| ITL (ms) | 1.52 | 1.39 | +8.90% | 2.53 | 2.38 | +6.11% |
| TPOT (ms) | 1.43 | 1.35 | +6.15% | 1.57 | 1.48 | +5.71% |
| E2E (ms) | 186.40 | 168.45 | +9.63% | 206.48 | 197.21 | +4.49% |

**怎么解读**：这条对 E2E 的收益非常大（P90 接近 10%），说明在这份 trace 下，“后半段 streaming/事件链路的背压”是实打实的大头；TTFT P99 反而略差，说明 TTFT 的尾巴主要不是事件 flush 能解决的那一类抖动。

**offline（vs inproc）**：`total tok/s -1.50%`（65314.13 → 64334.99）。这也符合预期：`aflush` 主要优化的是 online 的事件/streaming 链路，offline 吞吐不但吃不到收益，还可能因为额外 flush 检查而略有开销。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/aflush/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/aflush/offline_summary.md`

</details>

<details>
<summary><code>admit64</code>：限制每轮 admit 数（<code>--mp-max-admit-per-iter 64</code>）</summary>

**做什么**：mp 引擎每次从队列里拿新请求进 scheduler 时，设一个上限（避免单轮 admit 过多导致 host-side 处理时间爆掉）。

实现点：`rosellm/roseinfer/mp.py` 的 `max_admit = mp_max_admit_per_iter`（决定 `admit_limit`）。

**为什么可能会快**：它更像是“给 engine loop 加节流”：在 `scale=0.4` 的 burst 下，节流能减少单轮的 worst-case host 开销，从而改善 TTFT 的一部分尾巴；但节流也可能降低吞吐、增加排队。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.31 | +3.82% | 13.40 | 14.34 | -6.99% |
| ITL (ms) | 1.52 | 1.51 | +0.57% | 2.53 | 2.55 | -0.80% |
| TPOT (ms) | 1.43 | 1.44 | -0.37% | 1.57 | 1.60 | -1.91% |
| E2E (ms) | 186.40 | 186.56 | -0.09% | 206.48 | 205.45 | +0.50% |

**怎么解读**：它能改善 TTFT P90，但对 P99 反而负，这很符合“节流改善平均、但在极端 burst 下可能造成更长排队”的直觉。它更像是一条需要仔细调参/结合 workload 的工程开关。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/admit64/online_summary.md`

</details>

<details>
<summary><code>opool</code>：overlap 资源池（<code>--overlap-resource-pool</code>）</summary>

**做什么**：overlap 调度里会频繁分配 pinned CPU buffer、CUDA event 等资源；`opool` 把它们做成池，循环复用，减少分配/释放带来的 host 抖动。

实现点：`rosellm/roseinfer/engine.py` 的 `_pool_take_cpu_long/_pool_put_cpu_long/_pool_take_event/_pool_put_event`。

**为什么可能会快**：对 TTFT/E2E tail 来说，最敏感的经常是 host-side 的“小抖动”（分配、同步、队列竞争）。资源池本质上是把这些抖动变成“只发生在启动阶段的一次性成本”。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.49 | +2.13% | 13.40 | 14.01 | -4.51% |
| ITL (ms) | 1.52 | 1.51 | +0.54% | 2.53 | 2.55 | -0.56% |
| TPOT (ms) | 1.43 | 1.44 | -0.40% | 1.57 | 1.57 | +0.23% |
| E2E (ms) | 186.40 | 186.89 | -0.26% | 206.48 | 205.52 | +0.46% |

**怎么解读**：它对 P90 有点收益，但 tail 未必更好——说明这条 trace 的 P99 可能不是由“分配抖动”主导；或者资源池的收益被其它更大的抖动盖过去了。

**offline（vs inproc）**：`total tok/s -1.62%`（65314.13 → 64254.11）。offline 同样不太吃得到这类“减少 host 抖动”的好处（反而多一点管理开销）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/opool/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/opool/offline_summary.md`

</details>

<details>
<summary><code>pslots</code>：预分配 paged slot table（<code>--prealloc-paged-slots</code>）</summary>

**做什么**：paged KV 需要给每个 session 分配一个 slot，并在 global block table 里写入映射。默认 slot capacity 不够时会扩容/重新分配；`pslots` 在启动时直接按 `kv_cache_max_concurrency` 把 capacity 扩到位，减少运行时扩容抖动。

实现点：`rosellm/roseinfer/mp.py`/`rosellm/roseinfer/server.py` 在 build engine 后调用 `engine._ensure_paged_slot_capacity(kv_cache_max_concurrency)`。

**为什么可能会快**：它专门针对 “极端时刻触发一次扩容，直接把 P99 拉爆” 的风险。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 11.93 | -3.66% | 14.80 | 17.49 | -18.19% |
| ITL (ms) | 1.52 | 1.51 | +0.90% | 2.55 | 2.49 | +2.50% |
| TPOT (ms) | 1.44 | 1.45 | -0.59% | 1.58 | 1.61 | -1.68% |
| E2E (ms) | 186.81 | 186.81 | -0.00% | 207.18 | 212.95 | -2.78% |

**怎么解读**：这份数据里它是负的，说明当时的 base 配置里 slot 扩容可能不是主要 tail 来源；甚至预分配带来的额外内存/初始化成本在 `scale=0.4` 的运行形态下反而引入了抖动。这类“防止扩容”优化经常非常 workload-dependent：如果你本来就不会扩容，它就只剩成本。

**offline（vs inproc）**：`total tok/s -1.25%`（65314.13 → 64496.48）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pslots/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/pslots/offline_summary.md`

</details>

---

### 6) 其它 overlap/实验性开关（很多都属于“我踩过的坑”，图里已经很清晰）

这类开关在结果里经常呈现“ITL/TPOT 变好但 TTFT/E2E 爆炸”，原因一般是：它把某些同步点从显式的地方挪走了，形成隐形队列或乱序积压，tail 会被放大。

如果你想看我当时更细的踩坑过程，本文上半部分已经写了 `qdec` / `scfp` / `pmb8` 的故事线；这里就只把数字放出来做对照：

<details>
<summary><code>qdec</code>：+query decode（失败尝试，P99 大爆炸）</summary>

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.33 | 52.02 | -403.58% | 11.53 | 74.17 | -543.20% |
| ITL (ms) | 1.52 | 1.31 | +13.90% | 2.38 | 44.92 | -1788.18% |
| TPOT (ms) | 1.44 | 1.84 | -27.84% | 1.56 | 2.07 | -32.80% |
| E2E (ms) | 186.91 | 247.67 | -32.51% | 205.67 | 301.34 | -46.52% |

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/qdec/online_summary.md`

</details>

<details>
<summary><code>scfp</code>：+single-chunk fp（失败尝试，p99 outlier）</summary>

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.33 | 9.93 | +3.89% | 11.53 | 13.52 | -17.24% |
| ITL (ms) | 1.52 | 1.51 | +0.56% | 2.38 | 2.35 | +1.23% |
| TPOT (ms) | 1.44 | 1.43 | +0.29% | 1.56 | 1.55 | +0.38% |
| E2E (ms) | 186.91 | 185.72 | +0.64% | 205.67 | 211.29 | -2.73% |

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/scfp/online_summary.md`

</details>

<details>
<summary><code>pmb8</code>：prefill bsz8（失败尝试）</summary>

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.33 | 10.11 | +2.13% | 11.53 | 13.37 | -15.93% |
| ITL (ms) | 1.52 | 1.52 | +0.07% | 2.38 | 2.37 | +0.51% |
| TPOT (ms) | 1.44 | 1.44 | -0.02% | 1.56 | 1.54 | +0.78% |
| E2E (ms) | 186.91 | 187.15 | -0.13% | 205.67 | 204.98 | +0.34% |

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pmb8/online_summary.md`

</details>

---

### 7) lastlogits：为什么 offline 提升最大，但 online 反而不一定？

<details>
<summary><code>lastlogits</code>：prefill 直接给出 first token 所需 logits（<code>--prefill-last-logits</code>）</summary>

**做什么**：prefill 结束时，本来模型内部已经算出了最后一个位置的 hidden state；`lastlogits` 直接把这个位置的 logits 输出出来，用它采样第一个生成 token，从而避免 “prefill 完以后还要再做一次 decode forward 才能拿到第一个 token”。

实现点：`rosellm/roseinfer/engine.py` 中 `logits_pos = last_pos if prefill_last_logits else None`，并把 `logits_pos` 传给 `model(...)`。

**为什么 offline 会很赚**：offline benchmark 通常固定每个 prompt 都要生成 `output_len` 个 token。少一次 forward，相当于每条 request 少一次完整 decode step；吞吐直接吃到红利。

**为什么 online 可能不赚甚至负**：online 的 TTFT/E2E 里，首 token 的关键往往不是“少一次 forward”，而是：

- prefill 何时开始（队列/调度）
- first token 何时能被送出去（streaming/背压）
- 以及各种 host-side 抖动（CUDA graph、同步、事件队列等）

所以 lastlogits 省掉的那一次 decode，有可能被其它更大的抖动淹没；同时把“采样 first token”的逻辑塞回 prefill 路径也可能引入额外同步点。

**实际结果（scale=0.4 online）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 11.51 | 12.01 | -4.33% | 14.80 | 14.95 | -1.04% |
| ITL (ms) | 1.52 | 1.53 | -0.36% | 2.55 | 2.51 | +1.85% |
| TPOT (ms) | 1.44 | 1.44 | -0.43% | 1.58 | 1.59 | -0.32% |
| E2E (ms) | 186.81 | 187.52 | -0.38% | 207.18 | 209.74 | -1.23% |

**offline（vs inproc）**：`total tok/s +10.53%`（65314.13 → 72191.59）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/offline_summary.md`

</details>

---

### 8) eager prefill / overlap 细分开关：为什么有的能压 P99，有的会制造尾巴？

这一组开关更像是在调 overlap scheduler 的“先后顺序/扫描方式/公平性策略”。它们的共同点是：**不是省 GPU 算力，而是改变 prefill 何时被调度、prefill completion 何时被处理**；因此对 TTFT（尤其 P99）非常敏感。

<details>
<summary><code>peager</code>：eager prefill（实验性开关，TTFT P99 大幅改善）</summary>

**做什么**：让 prefill 更“积极”地被调度执行（summary 里标记为 `+eager prefill`）。你可以把它理解成：“prefill 不再被动地等 decode 间隙，而是更倾向于尽快开跑/尽快推进到 first token”。

> 注：这条开关在当时版本里是 overlap 调度的一个分支（命令行里叫 `--overlap-eager-prefill`）。当前 main 是否还保留同名开关不影响这份数据的解读。

**为什么会快**：TTFT = “排队等 prefill + prefill 本身 + first token 落地”。当负载很大时，“等 prefill” 往往是 TTFT tail 的主要来源之一；eager prefill 通过更激进的调度减少了这段等待，所以 TTFT P90/P99 都会明显下降。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 9.84 | +8.25% | 13.40 | 11.64 | +13.19% |
| ITL (ms) | 1.52 | 1.52 | +0.23% | 2.53 | 2.42 | +4.42% |
| TPOT (ms) | 1.43 | 1.44 | -0.56% | 1.57 | 1.58 | -0.12% |
| E2E (ms) | 186.40 | 186.19 | +0.11% | 206.48 | 204.87 | +0.78% |

**怎么解读**：这就是那种“专门压 TTFT tail 的调度开关”——TTFT P99 的收益非常扎实；对 decode 指标几乎不动（甚至略有 tradeoff，这是正常的：更偏首包优先时，稳态吞吐可能会小幅受影响）。

**offline（vs inproc）**：`total tok/s -1.51%`（65314.13 → 64328.54）。这也正常：offline 没有在线队列，eager 调度更多是一种额外控制逻辑，收益难以体现。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/peager/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/peager/offline_summary.md`

</details>

<details>
<summary><code>pfront</code>：prefill completion 优先（<code>+prefill complete first</code>）</summary>

**做什么**：当 overlap 模式下同时存在 prefill completion 和 decode completion 时，优先处理 prefill completion（summary 里标记为 `+prefill complete first`）。

**为什么可能会快**：prefill completion 通常意味着“这个请求离 first token 只差一步”，把它优先处理理论上能压 TTFT。

**为什么可能不快**：优先级策略如果打乱了整体的处理顺序，可能带来：

- completion 的“局部最优”但吞吐不一定更好；
- 更糟的是，某些请求的 completion 被反复推迟，造成 tail。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 9.84 | 9.96 | -1.21% | 11.64 | 11.99 | -3.03% |
| ITL (ms) | 1.52 | 1.51 | +0.26% | 2.42 | 2.41 | +0.52% |
| TPOT (ms) | 1.44 | 1.43 | +0.53% | 1.58 | 1.57 | +0.33% |
| E2E (ms) | 186.19 | 186.24 | -0.03% | 204.87 | 205.18 | -0.15% |

**怎么解读**：这条在这份数据里基本是负/接近噪声：说明 “completion 优先” 并没有解决 TTFT 的主要等待项，反而可能引入轻微的乱序与额外调度开销。

**offline（vs inproc）**：`total tok/s -1.38%`（65314.13 → 64415.37）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pfront/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/pfront/offline_summary.md`

</details>

<details>
<summary><code>pcap8</code>：prefill cap 8（限制 prefill 并发，典型负优化）</summary>

**做什么**：把某个 prefill 相关的并发上限（通常是 batch size）限制为 8（summary 里标记为 `+pcap8`）。

**为什么可能会快**：它的动机通常是“prefill 单次跑得更短，从而 TTFT 更稳”；代价是吞吐下降。

**为什么经常会慢**：在重载下，吞吐下降会导致队列更长，TTFT/E2E 的尾巴反而更糟。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 9.84 | 10.07 | -2.41% | 11.64 | 13.06 | -12.20% |
| ITL (ms) | 1.52 | 1.52 | -0.10% | 2.42 | 2.39 | +1.30% |
| TPOT (ms) | 1.44 | 1.44 | +0.01% | 1.58 | 1.60 | -1.45% |
| E2E (ms) | 186.19 | 187.15 | -0.51% | 204.87 | 212.46 | -3.70% |

**offline（vs inproc）**：`total tok/s -27.82%`（65314.13 → 47143.17）。这几乎是“把吞吐砍了一大刀”的信号，说明 cap 的副作用非常大。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/offline_summary.md`

</details>

<details>
<summary><code>pscan4</code>：prefill ready scan=4（<code>--overlap-prefill-ready-scan 4</code>）</summary>

**做什么**：在 overlap 模式下，为了“更快找到 ready 的 prefill 请求”，对队列做有限步数的扫描（这里是 4）。你可以把它理解成一种 “从 FIFO 变成带扫描/选择的队列策略”。

**为什么可能会快**：如果队列里有很多还没 ready 的请求，扫描能更快找到能推进的那批，减少空转。

**为什么可能会慢**：扫描/选择本身是额外开销；更重要的是，策略一旦不稳定就可能制造 tail（例如某些请求长期被跳过）。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 9.84 | 10.15 | -3.18% | 11.64 | 12.36 | -6.23% |
| ITL (ms) | 1.52 | 1.52 | -0.38% | 2.42 | 2.39 | +1.38% |
| TPOT (ms) | 1.44 | 1.44 | +0.05% | 1.58 | 1.57 | +0.55% |
| E2E (ms) | 186.19 | 187.13 | -0.51% | 204.87 | 204.88 | -0.01% |

**怎么解读**：这条更像是“调度策略引入了额外开销，但没有解决最主要的等待”，整体是负/接近噪声级别。

原始命令行可以在 `outputs/benchmarks/serving/online_merged_077_pscan4.json` 里看到（关键词 `--overlap-prefill-ready-scan`）。

</details>

---

### 9) stream / token buffer：为什么这类开关经常伤 TTFT tail？

这一类开关本质上在动 “token 如何从 engine 流到 server，再流到客户端” 的链路。它们通常对 **E2E/ITL** 有机会（背压更小），但如果实现引入了隐形队列、或者把同步点挪走了，就非常容易把 TTFT P99 拉爆。

<details>
<summary><code>pstream</code>：prefill 高优先级 stream（<code>--prefill-high-priority-stream</code>）</summary>

**做什么**：把 prefill 放到更高优先级的 CUDA stream 上跑，希望它在和 decode 竞争 SM 时更“抢得到”资源，从而减少 TTFT 等待。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.11 | +5.66% | 13.40 | 13.61 | -1.56% |
| ITL (ms) | 1.52 | 1.50 | +1.08% | 2.53 | 2.87 | -13.40% |
| TPOT (ms) | 1.43 | 1.44 | -0.38% | 1.57 | 1.58 | -0.41% |
| E2E (ms) | 186.40 | 186.16 | +0.13% | 206.48 | 205.95 | +0.26% |

**怎么解读**：P90 的 TTFT 确实更好了（符合“高优先级抢资源”的直觉），但 ITL P99 明显变差，说明它可能让 decode 的尾巴更不稳（stream 优先级改变了资源竞争，导致某些 decode batch 变成 outlier）。

**offline（vs inproc）**：`total tok/s +0.30%`（65314.13 → 65511.07），基本接近噪声。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/pstream/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/pstream/offline_summary.md`

</details>

<details>
<summary><code>tokbuf</code>：prefill token buffer（<code>--prefill-token-buffer</code>）</summary>

**做什么**：把 prefill 过程中产生/需要的 token（或其事件）先缓存在一块 buffer 里，再以更“批量”的方式交给后续链路处理。动机通常是减少频繁的小对象/小事件带来的开销。

**为什么可能会快**：减少事件数量/减少锁竞争/减少频繁的小拷贝。

**为什么可能会慢**：buffer 就是队列；如果 flush 时机不对，它会直接制造 TTFT tail（first token 被缓冲住，就算 GPU 已经算完也出不去）。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.72 | 10.68 | +0.41% | 13.40 | 15.19 | -13.32% |
| ITL (ms) | 1.52 | 1.53 | -0.28% | 2.53 | 2.58 | -1.97% |
| TPOT (ms) | 1.43 | 1.44 | -0.43% | 1.57 | 1.61 | -2.45% |
| E2E (ms) | 186.40 | 186.53 | -0.07% | 206.48 | 206.51 | -0.02% |

**怎么解读**：TTFT P99 明显变差，这很像 buffer 的 flush 时机/背压机制在极端 burst 下把 first token 卡住了。P90 变化很小，但 tail 被放大——这是线上最危险的一类负优化。

**offline（vs inproc）**：`total tok/s +0.66%`（65314.13 → 65748.01）。offline 的正收益并不能证明 online 安全：因为 offline 没有真实 streaming 首包。

原始命令行可以在 `outputs/benchmarks/serving/online_merged_077_tokbuf.json` 里看到（关键词 `--prefill-token-buffer`）。

</details>

<details>
<summary><code>tokbuf2</code>：token buffer + eager prefill + idleka（组合实验）</summary>

这条是把 token buffer 放到另一条 “peager+idleka” 的链路上，测它在不同上下文里的表现。

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.33 | 10.43 | -0.91% | 11.53 | 13.24 | -14.84% |
| ITL (ms) | 1.52 | 1.52 | +0.01% | 2.38 | 2.50 | -5.11% |
| TPOT (ms) | 1.44 | 1.44 | +0.03% | 1.56 | 1.61 | -3.56% |
| E2E (ms) | 186.91 | 186.86 | +0.02% | 205.67 | 205.54 | +0.07% |

**怎么解读**：结论一致：TTFT tail 依旧很差。说明它的风险与调度上下文无关，更像是 buffer 机制本身不适合在线首包链路。

**offline（vs inproc）**：`total tok/s -1.86%`（65314.13 → 64101.55）。

原始命令行可以在 `outputs/benchmarks/serving/online_merged_077_tokbuf2.json` 里看到。

</details>

---

### 10) idle keepalive：为什么它可能压掉 outlier，但不保证 P90 更好？

<details>
<summary><code>idleka</code>：mp idle keepalive（<code>--mp-idle-*</code>）</summary>

**做什么**：当 mp 引擎空闲时，不是完全“睡死”，而是用一种更温和的方式 keep CUDA 上下文/线程/状态活跃（避免从完全 idle 状态切回忙状态时的抖动）。

**为什么可能会快**：它本质上是在减少冷启动/唤醒成本，理论上对 P99 更友好。

**为什么可能不快**：keepalive 不是免费的：它可能增加常驻开销、影响系统的调度与 cache 行为，所以 P90 未必更好。

**实际结果（scale=0.4）**：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 9.84 | 10.33 | -5.02% | 11.64 | 11.53 | +0.90% |
| ITL (ms) | 1.52 | 1.52 | -0.38% | 2.42 | 2.38 | +1.71% |
| TPOT (ms) | 1.44 | 1.44 | +0.28% | 1.58 | 1.56 | +1.29% |
| E2E (ms) | 186.19 | 186.91 | -0.39% | 204.87 | 205.67 | -0.39% |

**怎么解读**：非常符合 “P99 更友好、P90 不一定更好” 的直觉：TTFT P99 略有改善，但 P90 反而差一点。对于线上来说，如果你只盯 tail，这类开关是有意义的；但它不一定是“普遍加速”。

**offline（vs inproc）**：`total tok/s -1.26%`（65314.13 → 64489.86）。

原始表/图：`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_summary.md`、`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/offline_summary.md`

</details>

---

### 11) fbsync：为什么它能同时改善 ITL/TPOT/E2E？

<details>
<summary><code>fbsync</code>：fast block table sync（<code>--fast-block-table-sync</code>）</summary>

**做什么**：paged KV cache 的 block table 需要在 CPU/GPU 之间同步（尤其 overlap/多进程下）。`fbsync` 是一次针对同步路径的优化：减少同步时的 Python 循环、减少拷贝次数/形状转换，目标是把这段 host-side 常数压到最小。

> 这条开关在当时版本里叫 `--fast-block-table-sync`（当前 main 是否保留同名开关不影响这份数据的解读）。

**为什么会快**：block table sync 属于那种“每步都要做、而且很容易被忽略的常数项”。它会同时影响：

- prefill：flashinfer paged prefill 需要 block table 才能正确 gather KV indices
- decode：paged attention decode 也需要 block table

所以同步路径一旦变快，ITL/TPOT/E2E 往往会一起变好。

**实际结果（scale=0.4）**（注意：它的 base/variant 在另一套基线上，所以收益幅度会比较大）：

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 15.66 | 15.25 | +2.57% | 26.17 | 24.46 | +6.55% |
| ITL (ms) | 1.54 | 1.38 | +9.83% | 2.73 | 2.33 | +14.60% |
| TPOT (ms) | 1.52 | 1.41 | +6.99% | 1.99 | 1.77 | +10.98% |
| E2E (ms) | 197.33 | 184.03 | +6.74% | 259.45 | 238.14 | +8.21% |

**怎么解读**：这就是那种“抠常数但抠到痛点上”的优化：对四个指标都是真收益，尤其 ITL/TPOT/E2E 的 tail 也一起被拉下来了。

**offline（vs inproc）**：`total tok/s -2.12%`（65314.13 → 63931.33）。offline 里这条反而是负的，说明它的收益主要来自在线多进程/同步/背压形态；offline 的纯吞吐里，它可能只是引入了额外控制逻辑或没有触发关键瓶颈。

原始命令行可以在 `outputs/benchmarks/serving/online_merged_077_fbsync.json` 里看到（关键词 `--fast-block-table-sync`）。

</details>

---

### 12) “同一个开关在不同上下文里”的两个例子：<code>pbuild2</code> / <code>lastlogits_peager</code>

这两条的意义是：**很多优化不是绝对的，它依赖你当前的调度/数据流上下文**。同一个想法放到另一条链路上，收益可能消失甚至变负。

<details>
<summary><code>pbuild2</code>：在 <code>peager+idleka</code> 链路上再开 <code>pbuild</code></summary>

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 10.33 | 10.60 | -2.64% | 11.53 | 13.76 | -19.33% |
| ITL (ms) | 1.52 | 1.52 | +0.47% | 2.38 | 2.38 | -0.01% |
| TPOT (ms) | 1.44 | 1.44 | +0.01% | 1.56 | 1.54 | +1.27% |
| E2E (ms) | 186.91 | 186.60 | +0.17% | 205.67 | 204.24 | +0.70% |

**怎么解读**：它对 E2E/TPOT 有一点点小收益，但 TTFT tail 明显变差——这很像 “pbuild 的 CPU 省时” 在这条链路里并不是瓶颈，反而额外分支/同步让 prefill 更容易出现 outlier。

**offline（vs inproc）**：`total tok/s -1.65%`（65314.13 → 64238.98）。

</details>

<details>
<summary><code>lastlogits_peager</code>：在 eager prefill 链路上再开 <code>lastlogits</code></summary>

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 9.84 | 10.16 | -3.27% | 11.64 | 12.43 | -6.84% |
| ITL (ms) | 1.52 | 1.51 | +0.19% | 2.42 | 2.35 | +2.73% |
| TPOT (ms) | 1.44 | 1.43 | +0.53% | 1.58 | 1.55 | +1.37% |
| E2E (ms) | 186.19 | 185.71 | +0.26% | 204.87 | 205.50 | -0.31% |

**怎么解读**：它改善了 ITL/TPOT（说明少一次 decode forward 的确有价值），但 TTFT 反而变差——这说明在 eager prefill 的调度里，“first token 到底怎么落地” 可能比 “少一次 forward” 更关键；把采样/同步挪进 prefill 路径会引入新的不稳定点。

</details>

---

### 13) evtq：为什么它会把 TTFT 直接炸穿？

<details>
<summary><code>evtq</code>：overlap query completions（极不稳定的负优化）</summary>

从结果看，这条属于“稳定性风险”而不是普通的负优化：P90/P99 都直接爆掉。

| metric | p90 base | p90 variant | p90 improve | p99 base | p99 variant | p99 improve |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 9.84 | 51.57 | -424.32% | 11.64 | 62.39 | -436.22% |
| ITL (ms) | 1.52 | 1.16 | +23.60% | 2.42 | 41.83 | -1628.16% |
| TPOT (ms) | 1.44 | 1.64 | -13.98% | 1.58 | 2.37 | -50.12% |
| E2E (ms) | 186.19 | 228.08 | -22.50% | 204.87 | 303.03 | -47.91% |

**为什么会出现“ITL P90 反而更好”的怪相**：ITL 的定义是 “first token 之后的 token 间隔”。当 TTFT 直接爆炸时，ITL 这段时间窗可能反而更“空”（比如 decode 频率变低、或统计样本形态发生改变），所以 ITL 的 P90 可能看起来更小，但这完全不代表系统更快。

**工程结论**：这类 “把 completion 变成查询驱动/事件驱动” 的策略，如果实现上存在隐形队列/乱序/同步点漂移，就会非常容易制造 TTFT tail。这个开关在当时结论就是：**不要开**。

原始命令行可以在 `outputs/benchmarks/serving/online_merged_077_evtq.json` 里看到（关键词 `--overlap-query-completions`）。

</details>

---

## 附录：Online/Offline 全量曲线图（不重新跑 benchmark）

这一节只做“把现有结果补齐成图集”：所有图片都来自 `outputs/` 里已有的 merged JSON 或历史 `figures_077_*`，**不重新跑 benchmark**。

<details>
<summary><code>admit64</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/admit64/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/admit64/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/admit64/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/admit64/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/admit64/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/admit64/offline_summary.md`

</details>

<details>
<summary><code>aflush</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/aflush/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/aflush/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/aflush/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/aflush/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/aflush/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/aflush/offline_summary.md`

</details>

<details>
<summary><code>bffwd</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/bffwd/offline_summary.md`

</details>

<details>
<summary><code>bt_noitem</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/bt_noitem/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/bt_noitem/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/bt_noitem/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/bt_noitem/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/bt_noitem/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/bt_noitem/offline_summary.md`

</details>

<details>
<summary><code>chunkbkt</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/chunkbkt/offline_summary.md`

</details>

<details>
<summary><code>dcap8</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/dcap8/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/dcap8/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/dcap8/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/dcap8/online_summary.md`

**Offline**：无现成数据（当时没有跑这个 feature 的 offline）。

</details>

<details>
<summary><code>evtq</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/evtq/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/evtq/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/evtq/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/evtq/online_summary.md`

**Offline**：无现成数据（当时没有跑这个 feature 的 offline）。

</details>

<details>
<summary><code>fanopast</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/fanopast/offline_summary.md`

</details>

<details>
<summary><code>fbsync</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/fbsync/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/fbsync/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/fbsync/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/fbsync/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/fbsync/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/fbsync/offline_summary.md`

</details>

<details>
<summary><code>idleka</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/idleka/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/idleka/offline_summary.md`

</details>

<details>
<summary><code>lastlogits</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/lastlogits/offline_summary.md`

</details>

<details>
<summary><code>nopad</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/nopad/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/nopad/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/nopad/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/nopad/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/nopad/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/nopad/offline_summary.md`

</details>

<details>
<summary><code>opool</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/opool/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/opool/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/opool/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/opool/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/opool/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/opool/offline_summary.md`

</details>

<details>
<summary><code>pbuild</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pbuild/offline_summary.md`

</details>

<details>
<summary><code>pcap8</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pcap8/offline_summary.md`

</details>

<details>
<summary><code>peager</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/peager/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/peager/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/peager/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/peager/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/peager/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/peager/offline_summary.md`

</details>

<details>
<summary><code>pfront</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pfront/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pfront/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pfront/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pfront/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pfront/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pfront/offline_summary.md`

</details>

<details>
<summary><code>pmb8</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pmb8/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pmb8/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pmb8/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pmb8/online_summary.md`

**Offline**：无现成数据（当时没有跑这个 feature 的 offline）。

</details>

<details>
<summary><code>pprio1</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pprio1/offline_summary.md`

</details>

<details>
<summary><code>pslots</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pslots/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pslots/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pslots/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pslots/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pslots/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pslots/offline_summary.md`

</details>

<details>
<summary><code>pstream</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pstream/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pstream/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/pstream/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pstream/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/pstream/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/pstream/offline_summary.md`

</details>

<details>
<summary><code>qdec</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/qdec/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/qdec/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/qdec/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/qdec/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/qdec/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/qdec/offline_summary.md`

</details>

<details>
<summary><code>rnopast</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/rnopast/offline_summary.md`

</details>

<details>
<summary><code>sbt</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/sbt/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/sbt/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/sbt/offline_summary.md`

</details>

<details>
<summary><code>scfp</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/scfp/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/scfp/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/scfp/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/scfp/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/scfp/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/scfp/offline_summary.md`

</details>

<details>
<summary><code>tokbuf</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/tokbuf/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/tokbuf/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/tokbuf/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/tokbuf/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/tokbuf/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/tokbuf/offline_summary.md`

</details>

<details>
<summary><code>wupcg16</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/online_summary.md`

**Offline**（吞吐）

![](/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/offline_throughput_compare.png)

- offline 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/wupcg16/offline_summary.md`

</details>

<details>
<summary><code>wuppf16</code></summary>

**Online**（TTFT/TPOT/ITL/E2E）

![](/assets/images/posts/2026-01-01-ttft-p99-077/wuppf16/online_latency_compare.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/wuppf16/online_latency_p99_only.png)

![](/assets/images/posts/2026-01-01-ttft-p99-077/wuppf16/online_latency_p90_only.png)

- online 原始表：`/assets/images/posts/2026-01-01-ttft-p99-077/wuppf16/online_summary.md`

**Offline**：无现成数据（当时没有跑这个 feature 的 offline）。

</details>
