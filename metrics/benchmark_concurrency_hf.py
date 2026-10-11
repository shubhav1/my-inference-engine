"""Same workload as benchmark_concurrency.py, but served by HuggingFace transformers' own
continuous batching engine (paged KV cache, no fixed slot count) instead of our chat.py
scheduler, for a like-for-like comparison of throughput, TTFT and latency at N concurrent chats.
Same dtype, device and KV memory budget as our engine.

    python metrics/benchmark_concurrency_hf.py "what changed since the last run"
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.configuration_utils import ContinuousBatchingConfig

from common import device_label, get_description
from benchmark_concurrency import (
    CONCURRENCY_LEVELS, KV_BYTES_PER_TOKEN, KV_RESERVED_MB, NO_EOS, REPLY_TOKENS, RESULTS_FILE, SEED,
    print_row, prompt_text, summarize, write_row,
)
from chat import DEVICE, DTYPE, MODEL_PATH, TEMPERATURE

# same KV memory budget as our paged cache, split into HF's own (larger) blocks
BLOCK_SIZE = 256
NUM_BLOCKS = int(KV_RESERVED_MB * 2**20) // (BLOCK_SIZE * KV_BYTES_PER_TOKEN)


def load_hf():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa")
    model.eval()
    return model.to(DEVICE), tokenizer


def start_manager(model):
    # same sampling recipe as our engine -- softmax at TEMPERATURE then multinomial, no
    # top-k/top-p truncation -- and no EOS, so every reply runs the full REPLY_TOKENS
    generation_config = GenerationConfig(
        do_sample=True, temperature=TEMPERATURE, top_k=0, top_p=1.0,
        max_new_tokens=REPLY_TOKENS, eos_token_id=NO_EOS,
    )
    cb_config = ContinuousBatchingConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS, seed=SEED)
    manager = model.init_continuous_batching(generation_config=generation_config, continuous_batching_config=cb_config)
    manager.start()
    return manager


def _run(manager, tokenizer, n, max_new_tokens):
    """Submit n requests at the same moment and block until every one has finished."""
    inputs = [tokenizer(prompt_text(i)).input_ids for i in range(n)]
    start = time.perf_counter()
    manager.add_requests(inputs, max_new_tokens=max_new_tokens, record_timestamps=True)
    outputs = []
    while len(outputs) < n:
        out = manager.get_result(timeout=300)
        if out is None:
            raise RuntimeError("HF continuous batching stopped returning results")
        if out.error is not None:
            raise RuntimeError(f"request {out.request_id} failed: {out.error}")
        if out.is_finished():
            outputs.append(out)
    return start, time.perf_counter() - start, outputs


def _peak_overlap(intervals):
    """Most intervals open at the same instant, i.e. how many requests were being served at once."""
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def run_level(manager, tokenizer, n):
    start, wall, outputs = _run(manager, tokenizer, n, REPLY_TOKENS)
    # every request was submitted at `start`, per-token timestamps share its perf_counter clock
    requests = [
        {"submit": start, "first": o.timestamps[0], "done": o.timestamps[-1], "tokens": len(o.generated_tokens)}
        for o in outputs
    ]
    peak_active = _peak_overlap([o.lifespan for o in outputs])
    return summarize(n, requests, wall, peak_active, kv_util=None)


def main():
    description = get_description()

    print("loading HF tokenizer and model...")
    model, tokenizer = load_hf()
    torch.manual_seed(SEED)
    manager = start_manager(model)

    print("warming up...")
    _run(manager, tokenizer, 1, 2)

    try:
        for n in CONCURRENCY_LEVELS:
            print(f"running {n} concurrent request(s) on {device_label()} (HuggingFace transformers)...")
            r = run_level(manager, tokenizer, n)
            print_row(r)
            write_row("HF transformers", description, r)  # append as each level finishes
    finally:
        manager.stop(block=True)

    print(f"results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
