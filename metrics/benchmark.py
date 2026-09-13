import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # chat.py/tokenizer.py live one level up

from chat import DEVICE, MODEL_PATH, generate, load_qwen
from tokenizer import QwenTokenizer

PROMPT = "Explain what a neural network is in simple terms."
NUM_TOKENS = 50
NO_EOS = -1  # no real token id is negative, so generate() never stops early and always runs NUM_TOKENS steps
RESULTS_FILE = Path(__file__).resolve().parent / "results.md"


def warmup(model, tokenizer):
    ids = tokenizer.encode(PROMPT)
    for _ in generate(model, None, ids, NO_EOS, max_tokens=1):
        pass


def run_benchmark(model, tokenizer):
    ids = tokenizer.encode(PROMPT)

    token_times = []
    start = time.perf_counter()
    prev = start
    for _ in generate(model, None, ids, NO_EOS, max_tokens=NUM_TOKENS):
        now = time.perf_counter()
        token_times.append(now - prev)
        prev = now
    total_time = time.perf_counter() - start

    ttft = token_times[0]
    tpot = sum(token_times[1:]) / (len(token_times) - 1) if len(token_times) > 1 else 0.0
    throughput = len(token_times) / total_time
    return ttft, tpot, throughput


def write_results(description, ttft, tpot, throughput):
    is_new_file = not os.path.exists(RESULTS_FILE)
    description = description.replace("|", "/").replace("\n", " ")
    now = datetime.now()

    with open(RESULTS_FILE, "a") as f:
        if is_new_file:
            f.write("# Benchmark Results\n\n")
            f.write("| Date | Time | Description | Device | Prompt | Tokens | TTFT (s) | TPOT (s) | Throughput (tok/s) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
        f.write(
            f"| {now.strftime('%Y-%m-%d')} | {now.strftime('%H:%M:%S')} | {description} | "
            f"{DEVICE.type} | {PROMPT} | {NUM_TOKENS} | {ttft:.3f} | {tpot:.3f} | {throughput:.2f} |\n"
        )


def main():
    description = input("Describe this run: ").strip()

    print("loading tokenizer and model...")
    tokenizer = QwenTokenizer(MODEL_PATH)
    model = load_qwen(MODEL_PATH)

    print("warming up...")
    warmup(model, tokenizer)

    print(f"benchmarking {NUM_TOKENS} tokens on {DEVICE.type}...")
    ttft, tpot, throughput = run_benchmark(model, tokenizer)

    print(f"time to first token: {ttft:.3f}s")
    print(f"time per output token: {tpot:.3f}s")
    print(f"throughput: {throughput:.2f} tokens/sec")

    write_results(description, ttft, tpot, throughput)
    print(f"results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
