"""Same benchmark as benchmark.py, but running the official HuggingFace transformers
implementation of Qwen2.5-1.5B instead of our own model.py for a good point of comparison."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # chat.py lives one level up
sys.path.insert(0, str(Path(__file__).resolve().parent))  # benchmark.py lives alongside this file

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import NUM_TOKENS, PROMPT, write_results
from chat import DEVICE, MODEL_PATH, TEMPERATURE

def load_hf():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.eval()
    return model.to(DEVICE), tokenizer


def generate(model, input_ids, max_tokens):
    """Mirrors chat.generate(): feed the prompt once, then one new token per step, using
    HF's own KV cache (past_key_values). Same sampling recipe as our engine -- softmax at
    TEMPERATURE, then multinomial, no top-k/top-p truncation -- for a like-for-like comparison.
    """
    next_ids = input_ids
    past_key_values = None
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(next_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        probs = torch.softmax(out.logits[0, -1] / TEMPERATURE, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_ids = next_id.unsqueeze(0)
        yield next_id.item()


def warmup(model, tokenizer):
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    for _ in generate(model, ids, max_tokens=1):
        pass


def run_benchmark(model, tokenizer):
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    token_times = []
    start = time.perf_counter()
    prev = start
    for _ in generate(model, ids, max_tokens=NUM_TOKENS):
        now = time.perf_counter()
        token_times.append(now - prev)
        prev = now
    total_time = time.perf_counter() - start

    ttft = token_times[0]
    tpot = sum(token_times[1:]) / (len(token_times) - 1) if len(token_times) > 1 else 0.0
    throughput = len(token_times) / total_time
    return ttft, tpot, throughput


def main():
    description = input("Describe this run: ").strip()

    print("loading HF tokenizer and model...")
    model, tokenizer = load_hf()

    print("warming up...")
    warmup(model, tokenizer)

    print(f"benchmarking {NUM_TOKENS} tokens on {DEVICE.type} (HuggingFace transformers)...")
    ttft, tpot, throughput = run_benchmark(model, tokenizer)

    print(f"time to first token: {ttft:.3f}s")
    print(f"time per output token: {tpot:.3f}s")
    print(f"throughput: {throughput:.2f} tokens/sec")

    write_results(f"[HF transformers] {description}", ttft, tpot, throughput)
    print(f"results appended to {Path(__file__).resolve().parent / 'results.md'}")


if __name__ == "__main__":
    main()
