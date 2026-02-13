# Offline Throughput Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- num_prompts: `128`
- input_len: `256`
- output_len: `64`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `173.73563841101713`
- run_start_time: `2025-12-28T18:31:30+08:00`
- run_end_time: `2025-12-28T18:34:24+08:00`
- versions: `git_rev=5099b2b, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=not installed, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| backend | req/s | output tok/s | total tok/s | total latency (s) |
|---|---:|---:|---:|---:|
| roseinfer | 201.13 | 12872.49 | 64362.44 | 0.636 |
| roseinfer (in-proc) | 204.11 | 13062.83 | 65314.13 | 0.627 |
| roseinfer (+kv256) | 0.00 | 0.00 | 0.00 | 8.936 |
| roseinfer (-affinity) | 327.87 | 20983.52 | 104917.61 | 0.390 |
| roseinfer (-batch send) | 328.31 | 21011.93 | 105059.64 | 0.390 |
| roseinfer (-cmd budg) | 327.52 | 20961.08 | 104805.40 | 0.391 |
| roseinfer (-fill tgt) | 327.34 | 20950.03 | 104750.13 | 0.391 |
| roseinfer (-thr cap) | 327.30 | 20947.42 | 104737.08 | 0.391 |
| roseinfer (+queue ipc) | 326.87 | 20919.91 | 104599.53 | 0.392 |
| roseinfer (-fast cnt) | 326.57 | 20900.67 | 104503.36 | 0.392 |
| roseinfer (+stream tok) | 206.20 | 13196.70 | 65983.50 | 0.621 |
| roseinfer (+warmup cg16) | 202.47 | 12957.92 | 64789.61 | 0.632 |
| SGLang | 238.44 | 15259.99 | 76299.94 | 0.537 |
| TensorRT-LLM | 0.00 | 0.00 | 0.00 | 23.099 |
| vLLM | 139.63 | 8936.27 | 44681.36 | 0.917 |
