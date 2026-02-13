# Offline Throughput Benchmark Summary
- model: `gpt2`
- dtype: `fp16`
- num_prompts: `128`
- input_len: `256`
- output_len: `64`
- sampling: temperature=0.7, top_p=0.9, top_k=50
- wall_s: `18.06826117899618`
- run_start_time: `2025-12-30T18:51:52+08:00`
- run_end_time: `2026-01-01T08:00:24+08:00`
- versions: `git_rev=a547c7c, rosellm=0.1.0, vllm=0.7.2, sglang=0.4.6, tensorrt_llm=1.1.0, torch=2.6.0, transformers=4.51.3, python=3.11.11`

| backend | req/s | output tok/s | total tok/s | total latency (s) |
|---|---:|---:|---:|---:|
| roseinfer | 201.13 | 12872.49 | 64362.44 | 0.636 |
| roseinfer (+idle keepalive) | 201.53 | 12897.97 | 64489.86 | 0.635 |
| roseinfer (in-proc) | 204.11 | 13062.83 | 65314.13 | 0.627 |
| roseinfer (+eager prefill) | 201.03 | 12865.71 | 64328.54 | 0.637 |
| roseinfer (+pprio1) | 203.70 | 13036.67 | 65183.33 | 0.628 |
| roseinfer (+tokbuf, +idle keepalive) | 200.32 | 12820.31 | 64101.55 | 0.639 |
| SGLang | 243.20 | 15564.48 | 77822.40 | 0.526 |
| TensorRT-LLM | 248.69 | 15916.24 | 79581.21 | 0.515 |
| vLLM | 140.44 | 8988.14 | 44940.70 | 0.911 |
