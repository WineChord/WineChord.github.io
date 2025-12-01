---
classes: wide2
title: "从零实现 LLM Training：015. Simple Generate"
excerpt: "在完成基础训练后，实现一个最简单的文本生成脚本。"
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

我们已经实现了张量并行、混合精度、checkpoint、eval and logging，并使用了真实数据做了一些小训练，现在我们可以搞一些简单的生成，来看看我们训练好的模型能吐出什么东西来。

## `generate.py`

```python
import argparse

import torch
from checkpoint import load_checkpoint
from config import GPTConfig
from dataset import build_tokenizer
from model import GPTModel


def top_k_logits(
    logits: torch.Tensor,  # [..., vocab]
    k: int,
) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)  # values.shape: [..., k]
    min_values = values[..., -1, None]  # min_values.shape: [..., 1]
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def top_p_logits(
    logits: torch.Tensor,  # [..., vocab]
    top_p: float,
) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )
    probs = torch.softmax(sorted_logits, dim=-1)  # [..., vocab]
    cum_probs = torch.cumsum(probs, dim=-1)  # [..., vocab]
    mask = cum_probs > top_p  # [..., vocab]
    mask[..., 0] = False  # keep at least one token
    masked_logits = sorted_logits.masked_fill(
        mask,
        float("-inf"),
    )
    _, original_indices = torch.sort(
        sorted_indices,
        dim=-1,
    )
    return torch.gather(masked_logits, -1, original_indices)


@torch.no_grad()
def generate(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    do_sample: bool = False,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()
    enc = tokenizer.encode(prompt, add_special_tokens=False)
    if len(enc) == 0:
        enc = [tokenizer.eos_token_id]
    input_ids = torch.tensor(
        [enc],
        dtype=torch.long,
        device=device,
    )
    max_pos = model.config.max_position_embeddings
    if input_ids.size(1) >= max_pos:
        input_ids = input_ids[:, -max_pos + 1 :]
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :]
        if not do_sample or temperature <= 0.0:
            next_token = torch.argmax(next_logits, dim=-1)
        else:
            logits_scaled = next_logits / temperature
            logits_filtered = logits_scaled
            if top_k > 0:
                logits_filtered = top_k_logits(logits_filtered, top_k)
            if top_p > 0.0:
                logits_filtered = top_p_logits(logits_filtered, top_p)
            probs = torch.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)[:, 0]
        input_ids = torch.cat(
            [input_ids, next_token.view(1, 1)],
            dim=1,
        )
        next_token_id = next_token.item()
        if (
            hasattr(tokenizer, "eos_token_id")
            and next_token_id == tokenizer.eos_token_id
        ):
            break
        if input_ids.size(1) > max_pos:
            input_ids = input_ids[:, -max_pos:]
    output_ids = input_ids[0].tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return text


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Using device: {device}")
    tokenizer = build_tokenizer(args.tokenizer_name)
    config = GPTConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_tensor_parallel=args.use_tensor_parallel,
    )
    model = GPTModel(config).to(device)
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    load_checkpoint(
        args.checkpoint_path,
        model=model,
        optimizer=None,
        scaler=None,
        map_location=device.type,
    )
    print(f"Prompt: {args.prompt}")
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=device,
    )
    print(f"Generated text: {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a model.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=10000,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of Transformer layers.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model hidden size.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=512,
        help="FFN hidden size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--use-tensor-parallel",
        action="store_true",
        help="Enable tensor parallel blocks.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="Tokenizer name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default="Hello, ",
        help="Prompt to generate text from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

```

在这个文件里面主要实现了 topk，topp，贪心等采样，目前简单起见，没有做 kv-cache。

这个时候我们可以把实际 gpt2 small 的配置拿出来，并且用一些大的数据集玩玩：

```shell
#!/bin/bash
torchrun --nproc_per_node=2 train_ddp.py \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.1 \
  --max-position-embeddings 1024 \
  --seq-len 1024 \
  --batch-size 2 \
  --num-steps 6000 \
  --lr 3e-4 \
  --train-data data/train.txt \
  --tokenizer-name gpt2 \
  --resume \
  --checkpoint-path checkpoints/gpt2_small_ddp.pt
```

这里的 `train.txt` 我手动放了几本鲁迅的书，从 gutenberg 下载然后转成简体中文的，大概前 4000 step 用 3e-4 的 lr，后 2000 用 1e-4 的 lr，由于数据量还是太小（1.3MB，大概 80w token），所以最后的 loss 可以很低，模型基本上记住了所有语料了。最后几个 step 的 log 见：

```shell
('epoch 10 step 5970 / 6000 ', 'train loss: 0.0641 ', 'val loss: 0.0100 ', 'val ppl: 1.0101 ', 'amp: True')
('epoch 10 step 5980 / 6000 ', 'train loss: 0.0660 ', 'val loss: 0.0100 ', 'val ppl: 1.0101 ', 'amp: True')
('epoch 10 step 5990 / 6000 ', 'train loss: 0.0469 ', 'val loss: 0.0102 ', 'val ppl: 1.0103 ', 'amp: True')
('epoch 10 step 6000 / 6000 ', 'train loss: 0.0482 ', 'val loss: 0.0100 ', 'val ppl: 1.0100 ', 'amp: True')
Training finished.
```

生成的话会发现其实是可以触发他去复述原文的：

```shell
$ cat generate_gpt2_small.sh 
#!/bin/bash
python generate.py \
  --checkpoint-path checkpoints/gpt2_small_ddp.pt \
  --tokenizer-name gpt2 \
  --vocab-size 50257 \
  --max-position-embeddings 1024 \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.0 \
  --prompt "我" \
  --max-new-tokens 500 \
  --temperature 0.99 \
  --top-p 0.99 \
  --do-sample \
  --device cuda
$ ./generate_gpt2_small.sh 
Using device: cuda
Loading checkpoint from checkpoints/gpt2_small_ddp.pt...
Prompt: 我
Generated text: 我要写下我的悔恨和悲哀，为子君，为自己。会馆〔２〕里的被遗忘在偏僻里的破屋是这样地寂静和空虚。时光过得真快，我爱子君，仗著她逃出这寂静和空虚，已经满一年了。事情又这么不凑巧，我重来时，偏偏空著的又只有这一间屋。依然是这样的破窗，这样的窗外的半枯的槐树和老紫藤，这样的窗前的方桌，这样的败壁，这样的靠壁的板床。深夜中独自躺在床上，就如我未曾和子君同居以前一般，过去一年中的时光全被消灭，全未有过，我并没有曾经从这破屋子搬出，在吉兆胡同创立了满怀希望的小小的家庭。不但如此。在一年之前，这寂静和空虚是并不这样的，常常含著期待﹔期待子君的到来。在久待的焦躁中，一听到皮鞋的高底尖触著砖路的清响，是怎样地使我骤然生动起来呵！于是就看见带著笑涡的苍白的圆脸，苍白的瘦的臂膊，布的有条纹的衫子，玄色的裙。她又带了窗外的半枯的槐树的新叶来，使我看见，还有挂在铁似的老干上的一房一房的紫白的藤花。然而现在呢，只有寂静和空虚依旧，子君却决不再来了，而且永远，永远地！……子君不在我这破屋里时，我什么也看不见。在百无聊赖中，顺手抓过一本书来，科学也好，
```

但是假如开头没见过，写的东西就不太对劲了：

```shell
$ ./generate_gpt2_small.sh 
Using device: cuda
Loading checkpoint from checkpoints/gpt2_small_ddp.pt...
Prompt: 我要为
Generated text: 我要为他的定将出。全似的是伙，而自己到了，我们并海裙租到来，不改了，改了，平改赵个爱肯的红慢的活无骂，没有什么东壁屁衣小之喊三旧，候，候，看了半到青年三岁进�裁爱。陈声，他的了他的意思，真站到他。笋意天，可是悲敲�有这一点太阳我的看见出下。那时出。他说︰“你们可看见过短，总要这实在是发明了！”“让是去！”他，一面著，来了蹊天便要这后来得意之后，说是不该过约要闲便只好，这么不倘来挽留快进门，连，这是可了子，知道我知道，这罢了。”“不渐渐的不是，息恰他说是饭也不再她便送鬼子吃一点的，还有料愿平没有留忘是四哩新兴开船春少，给价偏我了她门然是，这一大把如此，却著，屋……。”“还是取出来。──但她的，听说第一样那里孔书！”“那是的！””大家便又总吃老走的事。不放严天，用到大恐嘴里去了，便不自己不到呢，昏又怕来。这出去了。老女的一条嫂子打畜，给他心许多幸福坐角了，没有觉他说要将打果……白天他瘦，败，都带了。但虽然照例到五鸡比虽然看远。“唔，踉！”下非特疑于是头子竟著唉，都惊说是没有，虽然也大料翻关了却回，但他肆，头也无肯的起来如此，�
```

当然，由于我们现在的 dataset 的处理其实还太原始（比如没有多文档设计，并且是严格按 seq_len 分段切，无法以任意位置为起点等），我们后续再完善整个 Pipeline，并使用更多丰富的语料做训练看看效果。
