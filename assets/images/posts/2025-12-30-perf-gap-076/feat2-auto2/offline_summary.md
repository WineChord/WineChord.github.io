# Offline Throughput Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- num_prompts: `128`
- input_len: `256`
- output_len: `64`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `85.3827867680011`
- run_start_time: `2025-12-30T11:11:38+08:00`
- run_end_time: `2025-12-30T11:13:03+08:00`
- versions: `git_rev=0cd08cb, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=1.1.0, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| backend | req/s | output tok/s | total tok/s | total latency (s) |
|---|---:|---:|---:|---:|
| roseinfer | 184.02 | 11600.55 | 58710.07 | 0.696 |
| roseinfer (prefill auto2) | 136.00 | 8633.58 | 43448.40 | 0.941 |
| roseinfer (in-proc) | 122.96 | 7714.51 | 39191.16 | 1.041 |
| SGLang | 239.60 | 15334.57 | 76672.85 | 0.534 |
| TensorRT-LLM | 250.68 | 15939.89 | 80114.66 | 0.511 |
| vLLM | 141.90 | 9048.16 | 45373.82 | 0.902 |
