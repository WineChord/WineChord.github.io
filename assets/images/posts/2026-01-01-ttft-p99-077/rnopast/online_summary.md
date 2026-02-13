# Online Serving Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- n: `200`
- scales: `[0.4, 0.8, 1.6]`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `225.87478081299923`
- run_start_time: `2025-12-31T11:19:41+08:00`
- run_end_time: `2026-01-01T07:59:25+08:00`
- versions: `git_rev=a547c7c, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=1.1.0, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| scale | backend | TTFT p50/p90/p99 (ms) | TPOT p50/p90/p99 (ms) | ITL p50/p90/p99 (ms) | E2E p50/p90/p99 (ms) |
|---:|---|---:|---:|---:|---:|
| 0.40 | roseinfer (+chunk1024, +batch16, +pprio1, +warmup cg16, +prefill meta, +fast plan) | 8.00/10.72/13.40 | 1.34/1.43/1.57 | 1.28/1.52/2.53 | 172.87/186.40/206.48 |
| 0.40 | roseinfer (+chunk1024, +batch16, +pprio1, +warmup cg16, +prefill meta, +fast plan, +ragged no-past) | 7.87/10.44/15.62 | 1.33/1.44/1.61 | 1.27/1.52/2.59 | 172.44/186.02/206.98 |
| 0.40 | roseinfer (+chunk1024, +batch16, +warmup cg16, +prefill meta, +fast plan) | 8.06/11.51/14.80 | 1.33/1.44/1.58 | 1.27/1.52/2.55 | 172.45/186.81/207.18 |
| 0.40 | roseinfer (in-proc, +chunk1024, +batch16, +prefill meta, +fast plan) | 8.35/12.24/30.55 | 1.35/1.48/1.96 | 1.28/1.53/2.79 | 174.09/192.99/257.87 |
| 0.40 | SGLang | 7.80/9.39/14.89 | 1.10/1.21/1.35 | 1.07/1.28/2.85 | 144.13/156.65/170.35 |
| 0.40 | TensorRT-LLM | 5.82/6.30/8.03 | 1.40/1.44/1.90 | 1.40/1.53/2.62 | 182.62/187.42/193.03 |
| 0.40 | vLLM | 9.20/10.15/14.54 | 1.59/1.83/1.99 | 1.53/1.87/3.41 | 201.93/234.64/253.63 |
| 0.80 | roseinfer (+chunk1024, +batch16, +pprio1, +warmup cg16, +prefill meta, +fast plan) | 5.17/6.35/6.79 | 1.27/1.34/1.37 | 1.24/1.42/1.77 | 162.84/175.33/178.64 |
| 0.80 | roseinfer (+chunk1024, +batch16, +pprio1, +warmup cg16, +prefill meta, +fast plan, +ragged no-past) | 5.02/6.18/7.19 | 1.27/1.35/1.38 | 1.24/1.42/1.77 | 162.08/175.28/178.72 |
| 0.80 | roseinfer (+chunk1024, +batch16, +warmup cg16, +prefill meta, +fast plan) | 5.16/6.31/7.28 | 1.27/1.35/1.36 | 1.24/1.42/1.77 | 162.43/175.53/179.23 |
| 0.80 | roseinfer (in-proc, +chunk1024, +batch16, +prefill meta, +fast plan) | 4.00/5.15/5.45 | 1.28/1.36/1.38 | 1.25/1.42/2.14 | 162.59/175.32/179.33 |
| 0.80 | SGLang | 8.63/9.83/14.13 | 1.07/1.15/1.27 | 1.06/1.21/2.02 | 142.70/149.64/160.67 |
| 0.80 | TensorRT-LLM | 5.71/6.28/6.90 | 1.39/1.41/1.51 | 1.38/1.50/2.02 | 181.14/184.33/188.56 |
| 0.80 | vLLM | 9.19/10.36/10.99 | 1.45/1.66/1.78 | 1.42/1.69/2.38 | 186.06/211.76/232.81 |
| 1.60 | roseinfer (+chunk1024, +batch16, +pprio1, +warmup cg16, +prefill meta, +fast plan) | 5.21/5.92/6.60 | 1.26/1.34/1.43 | 1.23/1.39/1.69 | 161.50/172.72/185.28 |
| 1.60 | roseinfer (+chunk1024, +batch16, +pprio1, +warmup cg16, +prefill meta, +fast plan, +ragged no-past) | 5.13/5.90/6.63 | 1.26/1.34/1.41 | 1.23/1.39/1.68 | 160.94/172.16/183.90 |
| 1.60 | roseinfer (+chunk1024, +batch16, +warmup cg16, +prefill meta, +fast plan) | 5.29/6.20/6.63 | 1.26/1.33/1.44 | 1.23/1.40/1.70 | 161.71/173.39/188.51 |
| 1.60 | roseinfer (in-proc, +chunk1024, +batch16, +prefill meta, +fast plan) | 4.03/4.60/5.61 | 1.28/1.35/1.44 | 1.24/1.39/2.01 | 161.72/173.67/185.27 |
| 1.60 | SGLang | 8.87/9.98/14.93 | 1.06/1.16/1.34 | 1.05/1.20/2.02 | 142.46/151.47/166.25 |
| 1.60 | TensorRT-LLM | 5.88/6.43/7.02 | 1.38/1.41/1.53 | 1.38/1.50/1.86 | 180.76/184.17/199.04 |
| 1.60 | vLLM | 9.44/10.73/11.37 | 1.37/1.56/1.78 | 1.38/1.60/2.02 | 182.80/201.53/230.43 |
