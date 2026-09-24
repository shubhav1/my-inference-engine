# Concurrency Benchmark Results

| Date | Time | Engine | Description | Device | N chats | Reply tok | Peak active | Throughput (tok/s) | TTFT p50 (s) | TTFT p99 (s) | Latency p50 (s) | Latency p99 (s) | TPOT (s) | KV reserved (MB) | KV util (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-23 | 18:08:17 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 1 | 64 | 1 | 18.23 | 0.134 | 0.134 | 3.502 | 3.502 | 0.053 | 896 | 0.5 |
| 2026-09-23 | 18:08:22 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 2 | 64 | 2 | 24.39 | 0.283 | 0.368 | 5.185 | 5.238 | 0.078 | 896 | 1.0 |
| 2026-09-23 | 18:08:32 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 4 | 64 | 4 | 27.92 | 0.483 | 1.710 | 8.969 | 9.155 | 0.131 | 896 | 9.0 |
| 2026-09-23 | 18:08:49 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 8 | 64 | 8 | 29.30 | 1.837 | 3.728 | 16.954 | 17.457 | 0.242 | 896 | 17.9 |
| 2026-09-23 | 18:09:24 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 16 | 64 | 8 | 29.53 | 10.311 | 21.000 | 26.655 | 34.645 | 0.251 | 896 | 17.9 |
| 2026-09-23 | 18:09:37 | HF transformers | HF continuous batching, paged/sdpa, same 896MB KV budget | mps | 1 | 64 | 1 | 18.67 | 0.060 | 0.060 | 3.428 | 3.428 | 0.053 | 896 | – |
| 2026-09-23 | 18:09:41 | HF transformers | HF continuous batching, paged/sdpa, same 896MB KV budget | mps | 2 | 64 | 2 | 36.19 | 0.086 | 0.086 | 3.537 | 3.537 | 0.055 | 896 | – |
| 2026-09-23 | 18:09:46 | HF transformers | HF continuous batching, paged/sdpa, same 896MB KV budget | mps | 4 | 64 | 4 | 43.75 | 1.216 | 1.216 | 5.852 | 5.852 | 0.074 | 896 | – |
| 2026-09-23 | 18:09:57 | HF transformers | HF continuous batching, paged/sdpa, same 896MB KV budget | mps | 8 | 64 | 8 | 50.76 | 0.484 | 0.484 | 10.086 | 10.086 | 0.152 | 896 | – |
| 2026-09-23 | 18:10:07 | HF transformers | HF continuous batching, paged/sdpa, same 896MB KV budget | mps | 16 | 64 | 16 | 97.22 | 1.197 | 1.197 | 10.533 | 10.533 | 0.148 | 896 | – |
| 2026-09-23 | 18:19:11 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 1 | 64 | 1 | 18.75 | 0.150 | 0.150 | 3.401 | 3.401 | 0.052 | 896 | 0.5 |
| 2026-09-23 | 18:19:16 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 2 | 64 | 2 | 24.47 | 0.241 | 0.328 | 5.156 | 5.217 | 0.078 | 896 | 1.0 |
| 2026-09-23 | 18:19:26 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 4 | 64 | 4 | 27.76 | 0.537 | 1.790 | 8.975 | 9.204 | 0.130 | 896 | 9.0 |
| 2026-09-23 | 18:19:43 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 8 | 64 | 8 | 29.75 | 0.890 | 3.628 | 16.631 | 17.193 | 0.242 | 896 | 17.9 |
| 2026-09-23 | 18:20:07 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 16 | 64 | 16 | 42.52 | 1.686 | 8.310 | 22.538 | 24.049 | 0.311 | 896 | 35.5 |
