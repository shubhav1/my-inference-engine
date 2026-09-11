import os
import time
from datetime import datetime

import torch

from chat import DEVICE, MODEL_PATH, load_qwen
from tokenizer import QwenTokenizer

PROMPT = "Explain what a neural network is in simple terms."
NUM_TOKENS = 50
RESULTS_FILE = "results.md"


def step(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids)
    next_id = logits[0, -1].argmax().item()  # .item() forces a sync, so timing is accurate on mps
    return torch.cat([input_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)


def warmup(model, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)
    step(model, input_ids)


def run_benchmark(model, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)

    token_times = []
    start = time.perf_counter()
    for _ in range(NUM_TOKENS):
        token_start = time.perf_counter()
        input_ids = step(model, input_ids)
        token_times.append(time.perf_counter() - token_start)
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
