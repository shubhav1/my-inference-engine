"""Single-chat latency benchmark: one prompt, NUM_TOKENS generated tokens, reporting time to first
token, time per output token and throughput. benchmark_hf.py runs the same thing through HF transformers.

    python metrics/benchmark.py "what changed since the last run"
"""
import time
from pathlib import Path

from common import device_label, get_description, timestamp
from chat import DEVICE, MODEL_PATH, generate, load_qwen
from tokenizer import QwenTokenizer

PROMPT = "Explain what a neural network is in simple terms."
NUM_TOKENS = 50
NO_EOS = -1  # no real token id is negative, so generation never stops early and always runs NUM_TOKENS steps
RESULTS_FILE = Path(__file__).resolve().parent / "results.md"


def time_generation(tokens):
    """Drain a generator that yields one token per step, returning (TTFT, TPOT, throughput)."""
    token_times = []
    start = prev = time.perf_counter()
    for _ in tokens:
        now = time.perf_counter()
        token_times.append(now - prev)
        prev = now
    total_time = time.perf_counter() - start

    ttft = token_times[0]
    tpot = sum(token_times[1:]) / (len(token_times) - 1) if len(token_times) > 1 else 0.0
    return ttft, tpot, len(token_times) / total_time


def report(ttft, tpot, throughput):
    print(f"time to first token: {ttft:.3f}s")
    print(f"time per output token: {tpot:.3f}s")
    print(f"throughput: {throughput:.2f} tokens/sec")


def write_results(description, ttft, tpot, throughput):
    is_new_file = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if is_new_file:
            f.write("# Benchmark Results\n\n")
            f.write("| Date | Time | Description | Device | Prompt | Tokens | TTFT (s) | TPOT (s) | Throughput (tok/s) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
        f.write(
            f"| {timestamp()} | {description} | {device_label()} | {PROMPT} | {NUM_TOKENS} | "
            f"{ttft:.3f} | {tpot:.3f} | {throughput:.2f} |\n"
        )
    print(f"results appended to {RESULTS_FILE}")


def main():
    description = get_description()

    print("loading tokenizer and model...")
    tokenizer = QwenTokenizer(MODEL_PATH)
    model = load_qwen(MODEL_PATH)
    ids = tokenizer.encode(PROMPT)

    print("warming up...")
    for _ in generate(model, ids, NO_EOS, max_tokens=2):
        pass

    print(f"benchmarking {NUM_TOKENS} tokens on {device_label()}...")
    results = time_generation(generate(model, ids, NO_EOS, max_tokens=NUM_TOKENS))
    report(*results)
    write_results(description, *results)


if __name__ == "__main__":
    main()
