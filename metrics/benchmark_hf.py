"""Same benchmark as benchmark.py, but running the official HuggingFace transformers implementation
of Qwen2.5-1.5B (same dtype and device as our engine) instead of our own model.py, as a baseline.

    python metrics/benchmark_hf.py "what changed since the last run"
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import device_label, get_description
from benchmark import NUM_TOKENS, PROMPT, report, time_generation, write_results
from chat import DEVICE, DTYPE, MODEL_PATH, TEMPERATURE


def load_hf():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
    model.eval()
    return model.to(DEVICE), tokenizer


def generate(model, input_ids, max_tokens):
    """Mirrors chat.generate(): feed the prompt once, then one new token per step, using HF's own
    KV cache (past_key_values). Same sampling recipe as our engine -- softmax at TEMPERATURE, then
    multinomial, no top-k/top-p truncation -- for a like-for-like comparison.
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


def main():
    description = get_description()

    print("loading HF tokenizer and model...")
    model, tokenizer = load_hf()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    print("warming up...")
    for _ in generate(model, ids, max_tokens=2):
        pass

    print(f"benchmarking {NUM_TOKENS} tokens on {device_label()} (HuggingFace transformers)...")
    results = time_generation(generate(model, ids, max_tokens=NUM_TOKENS))
    report(*results)
    write_results(f"[HF transformers] {description}", *results)


if __name__ == "__main__":
    main()
