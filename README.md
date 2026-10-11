# my-inference-engine

An LLM inference engine that runs Qwen's pretrained **Qwen2.5-1.5B** weights on a hand-written implementation of the model architecture, with a paged KV cache, a continuous-batching scheduler, and a custom **Triton FlashAttention-2 kernel**. It's benchmarked against Hugging Face `transformers` on a cloud GPU.

## Highlights

- **Matches Hugging Face at low concurrency:** 99% of HF `transformers`' throughput with 1 chat and 98% with 2, on the same NVIDIA A10G, in bf16 with the same KV-cache memory budget.
- **Scales with concurrency:** throughput grows **~10×**, from 31.6 to 312 tokens/s, going from 1 to 16 concurrent chats.
- **Custom attention kernel:** a FlashAttention-2 implementation in Triton with native grouped-query attention, a different sequence length per row (ragged batches), and causal masking aligned to each row's end. It's checked against a PyTorch reference.
- **Profiled and optimized:** one round of profiling-driven fixes raised concurrent throughput **9–19%**. Earlier work on Apple Silicon took single-chat speed from 8.9 to 19.2 tokens/s (**2.2×**).

## What's inside

| Component | File | What it does |
|---|---|---|
| Model | [model.py](model.py) | Qwen2 architecture by hand: RoPE, SwiGLU, RMSNorm (fp32), grouped-query attention, tied embeddings |
| Tokenizer | [tokenizer.py](tokenizer.py) | Byte-level BPE tokenizer that reads Qwen's vocab and merges files |
| Paged KV cache | [kv_cache.py](kv_cache.py) | One preallocated pool split into 16-token blocks. Each sequence takes blocks only as it grows, with no worst-case reservation |
| Scheduler | [chat.py](chat.py) | Continuous batching: requests join and leave the running batch each step. Admission is based on free blocks, idle chats' caches are evicted when memory runs short, and sampling is batched with one GPU sync per step |
| Attention kernel | [kernels/fa2.py](kernels/fa2.py) | Triton FlashAttention-2 for prefill and decode, tile configs tuned on the A10G, PyTorch SDPA fallback off CUDA |
| Kernel tests | [kernels/test_fa2.py](kernels/test_fa2.py) | bf16 kernel vs fp32 SDPA reference across prefill, decode, and ragged batches |
| Web UI | [web.py](web.py) | FastAPI chat server that streams responses |
| Benchmarks | [metrics/](metrics/) | Single-chat and concurrent-chat benchmarks for this engine and for HF `transformers` |
| GPU runner | [modal_run.py](modal_run.py) | Runs the tests and benchmarks on a Modal A10G and copies the results back |

## Benchmarks

**Setup:** NVIDIA A10G (AWS, pinned for consistent hardware), bf16. Both engines use the same weights, the same sampling (temperature 1, no top-k/top-p), and the same 448 MB KV-cache budget. The baseline is HF `transformers`' own continuous-batching engine with SDPA attention.

### Concurrent chats

Each chat sends one prompt (lengths mixed from short to ~1,000 tokens) and gets a 64-token reply. All chats start at the same moment.

| Concurrent chats | This engine (tok/s) | HF transformers (tok/s) | Relative to HF |
|---|---|---|---|
| 1 | 31.6 | 32.0 | **99%** |
| 2 | 59.7 | 61.1 | **98%** |
| 4 | 113.5 | 120.6 | 94% |
| 8 | 199.7 | 246.1 | 81% |
| 16 | 312.4 | 459.7 | 68% |

Up to 4 chats, decode speed matches HF: 32–33 ms per token vs 31–32 ms. At 8 or more chats, the gap comes from prefill scheduling: new prompts are processed one at a time, while HF packs them into the running batch. That's the next item on the roadmap.

### Optimization log

| Change | Effect |
|---|---|
| KV cache, continuous batching, fused attention (Apple Silicon, fp32) | Single-chat throughput **8.9 → 19.2 tok/s** |
| FlashAttention-2 kernel with native GQA | Removes the 6× copy of K/V heads that grouped-query attention needed before every attention call |
| Ragged batches in the kernel (a different sequence length per row) | Concurrent chats at different lengths use FA2 instead of falling back to padded, masked SDPA |
| Logits for the last position only, batched sampling, no extra forward pass when a reply ends | Concurrent throughput **+9% to +19%** |

Full results history: [metrics/results.md](metrics/results.md) and [metrics/concurrency_results.md](metrics/concurrency_results.md).

## Running it

Place the Qwen2.5-1.5B weights in `./qwen2.5-1.5b/`, then:

```bash
uv sync
uv run web.py                                        # chat UI at http://127.0.0.1:5050
uv run metrics/benchmark.py "what changed"           # single-chat benchmark
uv run metrics/benchmark_concurrency.py "what changed"
```

On a cloud GPU via [Modal](https://modal.com):

```bash
uv run modal run modal_run.py::download_weights      # one-time: weights into a Modal volume
uv run modal run modal_run.py::test_fa2              # kernel correctness tests
uv run modal run modal_run.py::benchmark --description "what changed"
```

## Roadmap

- **Mixed prefill and decode batches:** pack new prompts and ongoing decode steps into one variable-length forward pass to close the time-to-first-token gap at high concurrency.
- **Paged attention inside the kernel:** read K/V straight from cache blocks instead of gathering a dense copy each step, with prefetching kept intact.
- **Decode-tuned kernel:** process all query heads that share a KV head in one tile, and capture decode steps as CUDA graphs.
