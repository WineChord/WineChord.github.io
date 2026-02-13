# Offline Throughput Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- num_prompts: `128`
- input_len: `256`
- output_len: `64`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `179.66707475704607`
- run_start_time: `2025-12-28T20:34:45+08:00`
- run_end_time: `2025-12-28T20:37:44+08:00`
- versions: `git_rev=5099b2b, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=not installed, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| backend | req/s | output tok/s | total tok/s | total latency (s) |
|---|---:|---:|---:|---:|
| roseinfer | 200.79 | 12850.40 | 64252.02 | 0.637 |
| roseinfer (in-proc) | 203.41 | 13018.42 | 65092.12 | 0.629 |
| roseinfer (kv conc=256) | 0.00 | 0.00 | 0.00 | 9.041 |
| roseinfer (no affinity split) | 201.12 | 12871.64 | 64358.18 | 0.636 |
| roseinfer (no batch send) | 198.65 | 12713.90 | 63569.49 | 0.644 |
| roseinfer (no cmd budget) | 199.67 | 12778.77 | 63893.85 | 0.641 |
| roseinfer (no fill target) | 201.09 | 12870.03 | 64350.13 | 0.637 |
| roseinfer (no thread cap) | 200.54 | 12834.33 | 64171.67 | 0.638 |
| roseinfer (queue ipc) | 200.59 | 12838.04 | 64190.19 | 0.638 |
| roseinfer (no fast finish counts) | 200.58 | 12837.32 | 64186.58 | 0.638 |
| roseinfer (stream token events) | 199.77 | 12785.18 | 63925.92 | 0.641 |
| SGLang | 239.77 | 15345.00 | 76725.00 | 0.534 |
| TensorRT-LLM | 252.43 | 16155.50 | 80777.49 | 0.507 |
| vLLM | 140.79 | 9010.75 | 45053.76 | 0.909 |
