---
classes: wide2
title: "从零实现 LLM Inference：015. Simple Benchmark"
excerpt: "实现简单的 benchmark，对比不同实现的性能。"
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

在完成了 inference server 并集成 online scheduler 之后，我们需要初步做一个简单的 benchmark 来对比不同实现的性能，从而有一个最初的认知，之后进行进一步性能优化的时候形成对比。



## 代码变更

主要就是新增一个 benchmark_scheduler.py 文件，里面分别测试 naive、offline scheduler、online scheduler 这三种方法对应的 tokens/s 性能。

### `benchmark_scheduler.py`

```python
import argparse
import time
from typing import List

from .engine import InferenceEngine, OfflineScheduler, OnlineScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the scheduler",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        required=True,
        help="Tokenizer name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 AMP on CUDA instead of float16.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to generate text from",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Prompts to generate text from",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=16,
        help="Number of requests to benchmark",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["naive", "online", "offline", "all"],
        help="Mode to run the benchmark",
    )
    return parser.parse_args()


def build_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts is not None:
        return args.prompts
    return [args.prompt for _ in range(args.num_requests)]


def count_tokens(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def report_stats(
    mode: str,
    engine: InferenceEngine,
    prompts: List[str],
    outputs: List[str],
    elapsed: float,
) -> None:
    assert len(prompts) == len(outputs)
    tokenizer = engine.tokenizer
    prompt_tokens = sum(count_tokens(tokenizer, p) for p in prompts)
    completion_tokens = 0
    for p, out in zip(prompts, outputs):
        t_prompt = count_tokens(tokenizer, p)
        t_out = count_tokens(tokenizer, out)
        if t_out > t_prompt:
            completion_tokens += t_out - t_prompt
    total_tokens = prompt_tokens + completion_tokens
    if elapsed <= 0:
        elapsed = 1e-6
    print(f"=== {mode} ===")
    print(f"Requests: {len(prompts)}")
    print(f"Elapsed: {elapsed:.6f} seconds")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput (completion): {total_tokens / elapsed:.2f} tokens/s")
    print(f"Throughput (total): {total_tokens / elapsed:.2f} tokens/s")
    print()


def benchmark_naive(
    engine: InferenceEngine,
    prompts: List[str],
    args: argparse.Namespace,
) -> None:
    outputs: List[str] = []
    t0 = time.perf_counter()
    for p in prompts:
        text = engine.generate(
            prompt=p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=True,
            do_sample=args.do_sample,
        )
        outputs.append(text)
    t1 = time.perf_counter()
    report_stats("naive", engine, prompts, outputs, t1 - t0)


def benchmark_offline(
    engine: InferenceEngine,
    prompts: List[str],
    args: argparse.Namespace,
) -> None:
    scheduler = OfflineScheduler(engine)
    request_ids: List[int] = []
    for p in prompts:
        rid = scheduler.add_request(
            p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=True,
            do_sample=args.do_sample,
        )
        request_ids.append(rid)
    t0 = time.perf_counter()
    outputs_by_id = scheduler.run()
    t1 = time.perf_counter()
    outputs: List[str] = []
    for rid in request_ids:
        text = outputs_by_id[rid]
        outputs.append(text)
    report_stats("offline", engine, prompts, outputs, t1 - t0)


def benchmark_online(
    engine: InferenceEngine,
    prompts: List[str],
    args: argparse.Namespace,
) -> None:
    scheduler = OnlineScheduler(engine, max_batch_size=args.max_batch_size)
    request_ids: List[int] = []
    for p in prompts:
        rid = scheduler.add_request(
            p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=True,
            do_sample=args.do_sample,
        )
        request_ids.append(rid)
    t0 = time.perf_counter()
    while scheduler.has_unfinished():
        scheduler.step()
    t1 = time.perf_counter()
    outputs: List[str] = []
    for rid in request_ids:
        text = scheduler.get_response(rid)
        outputs.append(text)
    report_stats("online", engine, prompts, outputs, t1 - t0)


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
    )
    prompts = build_prompts(args)
    if args.mode in ("naive", "all"):
        benchmark_naive(engine, prompts, args)
    if args.mode in ("offline", "all"):
        benchmark_offline(engine, prompts, args)
    if args.mode in ("online", "all"):
        benchmark_online(engine, prompts, args)


if __name__ == "__main__":
    main()

```



## 运行

运行下这个 benchmark 看看结果，可以看到 offline 和 online 基本差不多，是 1k tokens/s，naive 的是 300 tokens/s，注意这里测的 --num-requests 还不能太大，因为目前 kv-block 里面的设计给每一个 layer 预留的最大 block 数是有限的：

```shell
$ python -m rosellm.roseinfer.benchmark_scheduler --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt --tokenizer-name gpt2 --device cuda --prompt "Hello, this is a benchmark prompt." --num-requests 8 --max-new-tokens 64 --temperature 0.7 --top
-p 0.9 --do-sample --mode all
=== naive ===
Requests: 8
Elapsed: 1.708230 seconds
Prompt tokens: 64
Completion tokens: 512
Total tokens: 576
Throughput (completion): 337.19 tokens/s
Throughput (total): 337.19 tokens/s

=== offline ===
Requests: 8
Elapsed: 0.547367 seconds
Prompt tokens: 64
Completion tokens: 512
Total tokens: 576
Throughput (completion): 1052.31 tokens/s
Throughput (total): 1052.31 tokens/s

=== online ===
Requests: 8
Elapsed: 0.539884 seconds
Prompt tokens: 64
Completion tokens: 512
Total tokens: 576
Throughput (completion): 1066.90 tokens/s
Throughput (total): 1066.90 tokens/s
```

