---
classes: wide2
title: "从零实现 LLM Inference：004. Batch"
excerpt: "实现 batch inference 功能，支持多条 prompt 同时推理。"
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

本 MR 来处理下 batch inference 功能，之前我们的推理引擎只支持单条 prompt，为了支持多条 prompt，我们需要支持 batch inference。

## 代码变更

### `engine.py`

添加了 `_encode_prompts_batch` `_sample_next_token_batch` `prefill_batch` `decode_step_batch` `generate_batch`

```python
class InferenceEngine:
  # ...
    def _encode_prompts_batch(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(prompts) > 0
        all_ids: list[list[int]] = []
        max_len = 0
        for text in prompts:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if not ids:
                ids = [self.eos_token_id]
            all_ids.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eos_token_id
        batch = []
        masks = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            batch.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        input_ids = torch.tensor(
            batch,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            masks,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = self._maybe_truncate(input_ids)
        if input_ids.size(1) < attention_mask.size(1):
            attention_mask = attention_mask[:, -input_ids.size(1) :]
        return input_ids, attention_mask

    def _sample_next_token_batch(
        self,
        logits: torch.Tensor,  # [..., batch, vocab]
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        batch_size = logits.size(0)
        next_ids = []
        for i in range(batch_size):
            next_id = self._sample_next_token(
                logits=logits[i : i + 1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            next_ids.append(next_id)
        return torch.tensor(
            next_ids,
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def prefill_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torch.amp import autocast

        input_ids = self._maybe_truncate(input_ids)
        if attention_mask is not None and input_ids.size(1) < attention_mask.size(1):
            attention_mask = attention_mask[:, -input_ids.size(1) :]
        if self.use_amp:
            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
            ):
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    past_key_values=None,
                    use_cache=True,
                )
        else:
            logits, _, presents = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                past_key_values=None,
                use_cache=True,
            )
        self.kv_cache = presents
        last_logits = logits[:, -1, :]  # [batch, vocab]
        return last_logits

    @torch.no_grad()
    def decode_step_batch(
        self,
        last_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_cache is not None
        from torch.amp import autocast

        input_ids = last_token_ids.view(-1, 1)  # [B, 1]
        if self.use_amp:
            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
            ):
                logits, _, presents = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=None,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                )
        else:
            logits, _, presents = self.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
        self.kv_cache = presents
        next_logits = logits[:, -1, :]  # [B, V]
        return next_logits  # [B, V]

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> list[str]:
        assert len(prompts) > 0
        self.model.eval()
        input_ids, attn_mask = self._encode_prompts_batch(prompts)
        batch_size = input_ids.size(0)
        last_logits = self.prefill_batch(
            input_ids,
            attention_mask=attn_mask,
        )
        generated_ids = [input_ids[b].tolist() for b in range(batch_size)]
        if max_new_tokens <= 0:
            outputs = []
            for ids in generated_ids:
                t = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=self.device,
                )
                text = self._decode_tokens(t)
                outputs.append(text)
            return outputs
        next_ids = self._sample_next_token_batch(
            logits=last_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        for b in range(batch_size):
            generated_ids[b].append(int(next_ids[b].item()))
        eos_flags = [False for _ in range(batch_size)]
        if stop_on_eos and self.eos_token_id is not None:
            for b in range(batch_size):
                if next_ids[b].item() == self.eos_token_id:
                    eos_flags[b] = True
        last_token_ids = next_ids
        for _ in range(max_new_tokens - 1):
            next_logits = self.decode_step_batch(last_token_ids)
            next_ids = self._sample_next_token_batch(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            for b in range(batch_size):
                token_id = int(next_ids[b].item())
                generated_ids[b].append(token_id)
                if stop_on_eos and self.eos_token_id is not None:
                    if not eos_flags[b] and token_id == self.eos_token_id:
                        eos_flags[b] = True
            last_token_ids = next_ids
        outputs: list[str] = []
        for b in range(batch_size):
            ids = generated_ids[b]
            if stop_on_eos and self.eos_token_id is not None:
                if self.eos_token_id in ids:
                    eos_pos = ids.index(self.eos_token_id)
                    ids = ids[: eos_pos + 1]
            t = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            )
            text = self._decode_tokens(t)
            outputs.append(text)
        return outputs

```



### `cli_generate_batch.py`

```python
import argparse

from .engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a model in batch mode",
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
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Prompts to generate text from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
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
        "--stop-on-eos",
        action="store_true",
        default=True,
        help="Stop on EOS token",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
    )
    print(f"[roseinfer-batch] device: {engine.device}")
    print(f"[roseinfer-batch] use_amp: {engine.use_amp}")
    print(f"[roseinfer-batch] prompts: {args.prompts}")
    outputs = engine.generate_batch(
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_on_eos=args.stop_on_eos,
        do_sample=args.do_sample,
    )
    for i, output in enumerate(outputs):
        print(f"[roseinfer-batch] output {i}: {output}")


if __name__ == "__main__":
    main()

```



## 运行

```shell
$ python -m roseinfer.cli_generate_batch --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp.pt --tokenizer-name gpt2 --prompts "hello," "hi," "I am" "You are" --max-new-tokens 64 --do-sample --temperature 0.8 --top-k 40 --top-p 0.95
[roseinfer-batch] device: cuda
[roseinfer-batch] use_amp: True
[roseinfer-batch] prompts: ['hello,', 'hi,', 'I am', 'You are']
[roseinfer-batch] output 0: hello, but is so that to make the
C.
The "The “What is it is in a long-181716s.
The name the main areas.
As they are also use.
This is not the best-D, the first to the two-1811.
C.
[roseinfer-batch] output 1: hi,
The early time of the first the new way and “What is a different different types.
The article,, or your child.
The same same different types of the same group, the other information on the world.
When it was found that that the environment, the development of the first to the
[roseinfer-batch] output 2: I am a way.
For a big, and the state of the the United States are the United States of the more, and
The first an way, the development and the early, the people have been found by the use a "t.
B-12.
The results of an object.
2,
[roseinfer-batch] output 3: You are a new water.
The American people, and the world for the long.
The American and the other of the water.
The name and the world,’s at the water and the same risk of this, and the world and, the “The main side, but of the world
For
```

