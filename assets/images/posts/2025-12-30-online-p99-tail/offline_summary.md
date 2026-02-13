# Offline Throughput Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- num_prompts: `128`
- input_len: `256`
- output_len: `64`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `178.8563782700803`
- run_start_time: `2025-12-29T00:11:08+08:00`
- run_end_time: `2025-12-29T00:14:07+08:00`
- versions: `git_rev=5099b2b, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=not installed, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| backend | req/s | output tok/s | total tok/s | total latency (s) |
|---|---:|---:|---:|---:|
| roseinfer | 201.14 | 12872.99 | 64364.95 | 0.636 |
| roseinfer (in-proc) | 204.01 | 13056.84 | 65284.22 | 0.627 |
| SGLang | 243.20 | 15564.48 | 77822.40 | 0.526 |
| TensorRT-LLM | 248.69 | 15916.24 | 79581.21 | 0.515 |
| vLLM | 140.44 | 8988.14 | 44940.70 | 0.911 |
