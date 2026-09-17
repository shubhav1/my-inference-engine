# Benchmark Results

| Date | Time | Description | Device | Prompt | Tokens | TTFT (s) | TPOT (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | 16:34:59 | initial run, dummy inference server | mps | Explain what a neural network is in simple terms. | 50 | 0.087 | 0.113 | 8.92 |
| 2026-09-12 | 21:53:22 | kv-cache implemented | mps | Explain what a neural network is in simple terms. | 50 | 0.231 | 0.057 | 16.42 |
| 2026-09-17 | 13:35:18 | continuous batching implemented (dynamic admission + padded batched decode steps) | mps | Explain what a neural network is in simple terms. | 50 | 0.102 | 0.058 | 16.98 |
