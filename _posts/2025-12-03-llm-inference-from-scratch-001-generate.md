---
classes: wide2
title: "从零实现 LLM Inference：001. Generate"
excerpt: "实现最基本的 greedy generate。"
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

在我们的训练框架实现了最基础功能的时候，我们不妨开一小部分推理框架，在推理框架的第一个 PR 中，我们仅实现最必要的 greedy generate，在之后的 PR 中我们会逐步加入 kv-cache 以及 top-k, top-p sampling 等能力。

## 代码变更

### `roseinfer/engine.py`

```python
from typing import Optional

import torch
from rosetrainer.config import GPTConfig
from rosetrainer.dataset import build_tokenizer
from rosetrainer.model import GPTModel


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_name: str = "gpt2",
        device: Optional[str] = None,
        use_amp: bool = True,
        max_position_embeddings: Optional[int] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        ckpt = torch.load(checkpoint_path, map_location=self.device.type)
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            print("cannot find config from checkpoints, use GPTConfig")
            config = GPTConfig()
        else:
            config = GPTConfig(**cfg_dict)
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings
        self.config = config
        self.model = GPTModel(config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.tokenizer = build_tokenizer(tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.config.vocab_size < self.tokenizer.vocab_size:
            raise ValueError("the model vocab_size is less than tokenizer vocab_size")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids  # [1, T0]

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def _maybe_truncate(self, input_ids: torch.Tensor) -> torch.Tensor:
        max_pos = self.config.max_position_embeddings
        if input_ids.size(1) > max_pos:
            input_ids = input_ids[:, -max_pos:]
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
    ) -> str:
        self.model.eval()
        input_ids = self._encode_prompt(prompt)  # [1, T0]
        input_ids = self._maybe_truncate(input_ids)  # [1, T]
        from torch.amp import autocast

        for _ in range(max_new_tokens):
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    logits, _ = self.model(  # [1, T, V]
                        input_ids=input_ids,
                        attention_mask=None,
                        labels=None,
                    )
            else:
                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                )
            next_logits = logits[:, -1, :]  # [1, V]
            next_token = torch.argmax(next_logits, dim=-1)  # [1]
            next_id = next_token.item()
            next_token_t = next_token.view(1, 1)  # [1, 1]
            input_ids = torch.cat(  # [1, T+1]
                [input_ids, next_token_t],
                dim=1,
            )
            input_ids = self._maybe_truncate(input_ids)
            if (
                stop_on_eos
                and self.eos_token_id is not None
                and next_id == self.eos_token_id
            ):
                break
        generated = input_ids[0]  # [T1]
        text = self._decode_tokens(generated)
        return text

```

核心逻辑是 generate，此时还没有涉及到 kv-cache 的使用。

### `roseinfer/cli_generate.py`

```python
import argparse

from .engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a model",
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
        "--prompt",
        type=str,
        required=True,
        help="Prompt to generate text from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
    )
    print(f"[roseinfer] device: {engine.device}")
    print(f"[roseinfer] use_amp: {engine.use_amp}")
    print(f"[roseinfer] prompt: {args.prompt}")
    output = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"[roseinfer] output: {output}")


if __name__ == "__main__":
    main()

```

## 运行

运行效果如下，随便选了一个当前训练中的 checkpoint 做的（在 edu_fineweb10b 训练了 30k 步的样子），在 greedy decoding 的时候有很明显的复读机问题：

```shell
m$ python -m roseinfer.cli_generate --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp.pt --tokenizer-name gpt2 --max-new-tokens 50 --prompt "What is freedom?"
[roseinfer] device: cuda
[roseinfer] use_amp: True
[roseinfer] prompt: What is freedom?
[roseinfer] output: What is freedom?
The most important thing is that the world is to be. The world is that the world is not the world is the world.
The world is the world’s largest. The world is the world’s largest.
The
```

我有个小尝试在这里临时改成了 top-k 采样，发现结果会好很多，不过正式的 top-k 采样还是等我们后面的 PR 吧！
