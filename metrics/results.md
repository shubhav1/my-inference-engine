# Benchmark Results

| Date | Time | Description | Device | Prompt | Tokens | TTFT (s) | TPOT (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11 | 20:34:07 | [HF transformers] A10G baseline, same dtype + KV budget | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.029 | 0.026 | 37.87 |
| 2026-09-11 | 16:34:59 | initial run, dummy inference server | mps | Explain what a neural network is in simple terms. | 50 | 0.087 | 0.113 | 8.92 |
| 2026-09-12 | 21:53:22 | kv-cache implemented | mps | Explain what a neural network is in simple terms. | 50 | 0.231 | 0.057 | 16.42 |
| 2026-09-17 | 13:35:18 | continuous batching implemented (dynamic admission + padded batched decode steps) | mps | Explain what a neural network is in simple terms. | 50 | 0.102 | 0.058 | 16.98 |
| 2026-09-19 | 12:40:10 | fused SDPA kernel restored + admission fairness | mps | Explain what a neural network is in simple terms. | 50 | 0.086 | 0.051 | 19.38 |
| 2026-09-20 | 22:08:13 | static preallocated KV cache | mps | Explain what a neural network is in simple terms. | 50 | 0.089 | 0.051 | 19.23 |
| 2026-10-11 | 20:23:13 | A10G: FA2 kernel + paged KV cache + continuous batching | cuda NVIDIA A10 bf16 | Explain what a neural network is in simple terms. | 50 | 0.024 | 0.022 | 44.99 |
| 2026-10-11 | 20:39:34 | A10G: FA2 per-row kv lengths (ragged batches use FA2) | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.030 | 0.029 | 34.97 |
| 2026-10-11 | 20:52:28 | A10G: FA2 reads K/V from paged cache (no dense copy) | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.029 | 0.027 | 36.70 |
| 2026-10-11 | 21:07:10 | A10G: step 1 + last-pos logits + batched sampling + no _finish forward | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.030 | 0.029 | 34.60 |
