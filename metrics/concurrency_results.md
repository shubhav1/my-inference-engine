# Concurrency Benchmark Results

| Date | Time | Engine | Description | Device | N chats | Reply tok | Peak active | Throughput (tok/s) | TTFT p50 (s) | TTFT p99 (s) | Latency p50 (s) | Latency p99 (s) | TPOT (s) | KV reserved (MB) | KV util (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-11 | 20:34:26 | HF transformers | A10G baseline, same dtype + KV budget | cuda NVIDIA A10G bf16 | 1 | 64 | 1 | 32.02 | 0.033 | 0.033 | 1.998 | 1.998 | 0.031 | 448 | – |
| 2026-10-11 | 20:34:28 | HF transformers | A10G baseline, same dtype + KV budget | cuda NVIDIA A10G bf16 | 2 | 64 | 2 | 61.08 | 0.136 | 0.136 | 2.095 | 2.095 | 0.031 | 448 | – |
| 2026-10-11 | 20:34:30 | HF transformers | A10G baseline, same dtype + KV budget | cuda NVIDIA A10G bf16 | 4 | 64 | 4 | 120.64 | 0.131 | 0.131 | 2.121 | 2.121 | 0.032 | 448 | – |
| 2026-10-11 | 20:34:32 | HF transformers | A10G baseline, same dtype + KV budget | cuda NVIDIA A10G bf16 | 8 | 64 | 8 | 246.05 | 0.051 | 0.051 | 2.080 | 2.080 | 0.032 | 448 | – |
| 2026-10-11 | 20:34:34 | HF transformers | A10G baseline, same dtype + KV budget | cuda NVIDIA A10G bf16 | 16 | 64 | 16 | 459.72 | 0.101 | 0.101 | 2.227 | 2.227 | 0.034 | 448 | – |
| 2026-09-23 | 18:08:17 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 1 | 64 | 1 | 18.23 | 0.134 | 0.134 | 3.502 | 3.502 | 0.053 | 896 | 0.5 |
| 2026-09-23 | 18:08:22 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 2 | 64 | 2 | 24.39 | 0.283 | 0.368 | 5.185 | 5.238 | 0.078 | 896 | 1.0 |
| 2026-09-23 | 18:08:32 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 4 | 64 | 4 | 27.92 | 0.483 | 1.710 | 8.969 | 9.155 | 0.131 | 896 | 9.0 |
| 2026-09-23 | 18:08:49 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 8 | 64 | 8 | 29.30 | 1.837 | 3.728 | 16.954 | 17.457 | 0.242 | 896 | 17.9 |
| 2026-09-23 | 18:09:24 | ours | static KV cache (8 slots), mixed prompt lengths | mps | 16 | 64 | 8 | 29.53 | 10.311 | 21.000 | 26.655 | 34.645 | 0.251 | 896 | 17.9 |
| 2026-09-23 | 18:19:11 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 1 | 64 | 1 | 18.75 | 0.150 | 0.150 | 3.401 | 3.401 | 0.052 | 896 | 0.5 |
| 2026-09-23 | 18:19:16 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 2 | 64 | 2 | 24.47 | 0.241 | 0.328 | 5.156 | 5.217 | 0.078 | 896 | 1.0 |
| 2026-09-23 | 18:19:26 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 4 | 64 | 4 | 27.76 | 0.537 | 1.790 | 8.975 | 9.204 | 0.130 | 896 | 9.0 |
| 2026-09-23 | 18:19:43 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 8 | 64 | 8 | 29.75 | 0.890 | 3.628 | 16.631 | 17.193 | 0.242 | 896 | 17.9 |
| 2026-09-23 | 18:20:07 | ours | paged KV cache (16-tok blocks, same 896MB budget) | mps | 16 | 64 | 16 | 42.52 | 1.686 | 8.310 | 22.538 | 24.049 | 0.311 | 896 | 35.5 |
| 2026-10-11 | 20:23:38 | ours | A10G: FA2 kernel + paged KV cache + continuous batching | cuda NVIDIA A10 bf16 | 1 | 64 | 1 | 42.37 | 0.025 | 0.025 | 1.487 | 1.487 | 0.023 | 448 | 0.5 |
| 2026-10-11 | 20:23:40 | ours | A10G: FA2 kernel + paged KV cache + continuous batching | cuda NVIDIA A10 bf16 | 2 | 64 | 2 | 80.16 | 0.050 | 0.076 | 1.551 | 1.573 | 0.024 | 448 | 1.0 |
| 2026-10-11 | 20:23:42 | ours | A10G: FA2 kernel + paged KV cache + continuous batching | cuda NVIDIA A10 bf16 | 4 | 64 | 4 | 142.44 | 0.100 | 0.207 | 1.705 | 1.772 | 0.025 | 448 | 9.0 |
| 2026-10-11 | 20:23:44 | ours | A10G: FA2 kernel + paged KV cache + continuous batching | cuda NVIDIA A10 bf16 | 8 | 64 | 8 | 232.58 | 0.221 | 0.465 | 2.016 | 2.175 | 0.028 | 448 | 17.9 |
| 2026-10-11 | 20:23:47 | ours | A10G: FA2 kernel + paged KV cache + continuous batching | cuda NVIDIA A10 bf16 | 16 | 64 | 16 | 314.87 | 0.467 | 1.004 | 2.873 | 3.219 | 0.038 | 448 | 35.5 |
| 2026-10-11 | 20:39:53 | ours | A10G: FA2 per-row kv lengths (ragged batches use FA2) | cuda NVIDIA A10G bf16 | 1 | 64 | 1 | 27.99 | 0.030 | 0.030 | 2.251 | 2.251 | 0.035 | 448 | 0.5 |
| 2026-10-11 | 20:39:55 | ours | A10G: FA2 per-row kv lengths (ragged batches use FA2) | cuda NVIDIA A10G bf16 | 2 | 64 | 2 | 54.95 | 0.120 | 0.198 | 2.266 | 2.296 | 0.034 | 448 | 1.0 |
| 2026-10-11 | 20:39:58 | ours | A10G: FA2 per-row kv lengths (ragged batches use FA2) | cuda NVIDIA A10G bf16 | 4 | 64 | 4 | 102.27 | 0.148 | 0.298 | 2.380 | 2.467 | 0.035 | 448 | 9.0 |
| 2026-10-11 | 20:40:00 | ours | A10G: FA2 per-row kv lengths (ragged batches use FA2) | cuda NVIDIA A10G bf16 | 8 | 64 | 8 | 177.06 | 0.298 | 0.608 | 2.644 | 2.855 | 0.037 | 448 | 17.9 |
| 2026-10-11 | 20:40:04 | ours | A10G: FA2 per-row kv lengths (ragged batches use FA2) | cuda NVIDIA A10G bf16 | 16 | 64 | 16 | 262.19 | 0.630 | 1.288 | 3.419 | 3.864 | 0.044 | 448 | 35.5 |
| 2026-10-11 | 20:52:48 | ours | A10G: FA2 reads K/V from paged cache (no dense copy) | cuda NVIDIA A10G bf16 | 1 | 64 | 1 | 33.79 | 0.031 | 0.031 | 1.863 | 1.863 | 0.029 | 448 | 0.5 |
| 2026-10-11 | 20:52:50 | ours | A10G: FA2 reads K/V from paged cache (no dense copy) | cuda NVIDIA A10G bf16 | 2 | 64 | 2 | 60.88 | 0.107 | 0.182 | 2.042 | 2.072 | 0.031 | 448 | 1.0 |
| 2026-10-11 | 20:52:52 | ours | A10G: FA2 reads K/V from paged cache (no dense copy) | cuda NVIDIA A10G bf16 | 4 | 64 | 4 | 105.45 | 0.151 | 0.304 | 2.299 | 2.391 | 0.034 | 448 | 9.0 |
| 2026-10-11 | 20:52:55 | ours | A10G: FA2 reads K/V from paged cache (no dense copy) | cuda NVIDIA A10G bf16 | 8 | 64 | 8 | 172.46 | 0.306 | 0.612 | 2.716 | 2.929 | 0.038 | 448 | 17.9 |
| 2026-10-11 | 20:53:00 | ours | A10G: FA2 reads K/V from paged cache (no dense copy) | cuda NVIDIA A10G bf16 | 16 | 64 | 16 | 240.40 | 0.780 | 1.455 | 3.753 | 4.208 | 0.047 | 448 | 35.5 |
| 2026-10-11 | 21:07:29 | ours | A10G: step 1 + last-pos logits + batched sampling + no _finish forward | cuda NVIDIA A10G bf16 | 1 | 64 | 1 | 31.63 | 0.033 | 0.033 | 2.022 | 2.022 | 0.032 | 448 | 0.5 |
| 2026-10-11 | 21:07:31 | ours | A10G: step 1 + last-pos logits + batched sampling + no _finish forward | cuda NVIDIA A10G bf16 | 2 | 64 | 2 | 59.74 | 0.097 | 0.160 | 2.125 | 2.141 | 0.032 | 448 | 1.0 |
| 2026-10-11 | 21:07:33 | ours | A10G: step 1 + last-pos logits + batched sampling + no _finish forward | cuda NVIDIA A10G bf16 | 4 | 64 | 4 | 113.46 | 0.142 | 0.280 | 2.209 | 2.253 | 0.033 | 448 | 9.0 |
| 2026-10-11 | 21:07:36 | ours | A10G: step 1 + last-pos logits + batched sampling + no _finish forward | cuda NVIDIA A10G bf16 | 8 | 64 | 8 | 199.70 | 0.284 | 0.570 | 2.451 | 2.557 | 0.034 | 448 | 17.9 |
| 2026-10-11 | 21:07:39 | ours | A10G: step 1 + last-pos logits + batched sampling + no _finish forward | cuda NVIDIA A10G bf16 | 16 | 64 | 16 | 312.37 | 0.641 | 1.256 | 3.037 | 3.267 | 0.038 | 448 | 35.6 |
