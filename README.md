# my-inference-engine

A from-scratch inference engine for Qwen2.5-1.5B, written in PyTorch with a custom Triton FlashAttention-2 kernel.

## Features

- **Model** ([model.py](model.py)): the Qwen2 architecture implemented by hand (RoPE, SwiGLU, RMSNorm, GQA, tied embeddings).
- **Paged KV cache** ([kv_cache.py](kv_cache.py)): one preallocated pool split into 16-token blocks. Each sequence takes blocks only as it grows.
- **Continuous batching** ([chat.py](chat.py)): requests join and leave the running batch dynamically. A request is admitted when enough blocks are free, and idle chats can be evicted.
- **FlashAttention-2** ([kernels/fa2.py](kernels/fa2.py)): an autotuned Triton kernel used for prefill and decode on CUDA, with SDPA as the fallback.
- **Web chat UI** ([web.py](web.py)): a FastAPI server that streams responses.

## Results

Single-chat benchmark, 50 tokens, compared with the first bare-bones version ([metrics/results.md](metrics/results.md)):

| Metric | Baseline | Current | Change |
|---|---|---|---|
| Throughput (tok/s) | 8.92 | 19.23 | **+116%** |
| Time per output token (s) | 0.113 | 0.051 | **−55%** |
| Time to first token (s) | 0.087 | 0.089 | +2% |

Concurrency results compared with HF transformers are in [metrics/concurrency_results.md](metrics/concurrency_results.md).

## Usage

Place the model weights in `./qwen2.5-1.5b/`, then:

```bash
uv sync
uv run web.py                              # chat UI at http://127.0.0.1:5050
uv run metrics/benchmark.py                # single-chat benchmark
uv run metrics/benchmark_concurrency.py    # concurrent-chat benchmark
```
