# Benchmark Results

| Date | Time | Description | Device | Prompt | Tokens | TTFT (s) | TPOT (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 2026-10-11 | 13:34:07 | [HF transformers] baseline: same GPU, dtype and KV budget | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.029 | 0.026 | 37.87 |
| 2026-09-11 | 16:34:59 | initial run, dummy inference server | mps fp32 | Explain what a neural network is in simple terms. | 50 | 0.087 | 0.113 | 8.92 |
| 2026-09-12 | 21:53:22 | kv-cache implemented | mps fp32 | Explain what a neural network is in simple terms. | 50 | 0.231 | 0.057 | 16.42 |
| 2026-09-17 | 13:35:18 | continuous batching implemented (dynamic admission + padded batched decode steps) | mps fp32 | Explain what a neural network is in simple terms. | 50 | 0.102 | 0.058 | 16.98 |
| 2026-09-19 | 12:40:10 | fused SDPA kernel restored + admission fairness | mps fp32 | Explain what a neural network is in simple terms. | 50 | 0.086 | 0.051 | 19.38 |
| 2026-09-20 | 22:08:13 | static preallocated KV cache | mps fp32 | Explain what a neural network is in simple terms. | 50 | 0.089 | 0.051 | 19.23 |
| 2026-10-11 | 13:39:34 | FA2 kernel: native GQA, per-row lengths for ragged batches | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.030 | 0.029 | 34.97 |
| 2026-10-11 | 13:52:28 | experiment (reverted): FA2 reads K/V straight from the paged cache | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.029 | 0.027 | 36.70 |
| 2026-10-11 | 14:07:10 | last-position logits only, batched sampling, no extra pass when a reply ends | cuda NVIDIA A10G bf16 | Explain what a neural network is in simple terms. | 50 | 0.030 | 0.029 | 34.60 |
