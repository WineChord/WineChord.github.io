# Offline Throughput Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- num_prompts: `128`
- input_len: `256`
- output_len: `64`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `136.63317050901242`
- run_start_time: `2025-12-27T21:48:36+08:00`
- run_end_time: `2025-12-27T21:50:53+08:00`
- versions: `git_rev=5099b2b, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=not installed, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| backend | req/s | output tok/s | total tok/s | total latency (s) |
|---|---:|---:|---:|---:|
| roseinfer | 103.10 | 6583.40 | 32978.22 | 1.241 |
| roseinfer (in-proc) | 92.50 | 5727.63 | 29407.05 | 1.384 |
| roseinfer (+kv256) | 112.63 | 7208.61 | 36043.04 | 1.136 |
| roseinfer (-affinity) | 81.27 | 5083.28 | 25888.78 | 1.575 |
| roseinfer (-cmd budg) | 98.37 | 6168.61 | 31350.27 | 1.301 |
| roseinfer (-thr cap) | 112.28 | 7185.68 | 35928.42 | 1.140 |
| roseinfer (+queue ipc) | 81.51 | 5109.23 | 25976.63 | 1.570 |
| roseinfer (-fast cnt) | 92.11 | 5821.44 | 29400.80 | 1.390 |
| roseinfer (+stream tok) | 98.59 | 6281.88 | 31520.30 | 1.298 |
| SGLang | 241.79 | 15474.87 | 77374.35 | 0.529 |
| vLLM | 143.14 | 9127.68 | 45772.58 | 0.894 |
