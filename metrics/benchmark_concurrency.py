"""Concurrency benchmark: drives the real scheduler in chat.py (admission, deferral, batched decode)
with N chats that all send their first message at the same moment, then measures aggregate
throughput, latency and how much of the KV cache is actually in use. Sweeping N up shows where
chats start queuing for cache capacity rather than running concurrently.
benchmark_concurrency_hf.py runs the same workload through HF transformers' continuous batching.

    python metrics/benchmark_concurrency.py "what changed since the last run"
"""
import statistics
import threading
import time
from pathlib import Path

import torch

from common import device_label, get_description, timestamp
import chat as engine
from chat import KV_CACHE, MODEL_PATH, Chat, load_qwen
from tokenizer import QwenTokenizer

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
REPLY_TOKENS = 64  # every reply runs exactly this many tokens (EOS and stop sequences disabled)
NO_EOS = -1  # no real token id is negative, so a reply never ends early on EOS
SEED = 0
SAMPLE_INTERVAL = 0.01  # seconds between samples of scheduler / cache occupancy
RESULTS_FILE = Path(__file__).resolve().parent / "concurrency_results.md"

# KV cache memory budget both engines get: whatever our paged cache's pool reserves
KV_BYTES_PER_TOKEN = KV_CACHE.buf[:, :, 0].numel() * KV_CACHE.buf.element_size()
KV_RESERVED_MB = KV_CACHE.buf.numel() * KV_CACHE.buf.element_size() / 2**20

_PARAGRAPH = (
    "Neural networks are built from layers of simple units that each compute a weighted sum of "
    "their inputs followed by a nonlinearity. Training adjusts the weights with gradient descent "
    "so that the network's outputs move closer to the desired targets. "
)

# mixed prompt lengths (short / medium / long) so the batch is ragged, chat i gets PROMPTS[i % 4]
PROMPTS = [
    "Explain what a neural network is in simple terms.",
    "Give me three tips for writing clean Python code, with a short example for each.",
    _PARAGRAPH * 6 + "Summarize the text above in two sentences.",
    _PARAGRAPH * 20 + "What are the main ideas described above?",
]


def prompt_text(i):
    """The exact text chat i sends, in the same turn format chat.py wraps every message in."""
    return f"User: {PROMPTS[i % len(PROMPTS)]}\nAssistant:"


def _percentile(values, pct):
    """Linear-interpolated percentile (numpy's default), with few samples p99 sits near the max."""
    ordered = sorted(values)
    pos = pct / 100 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(n, requests, wall, peak_active, kv_util):
    """requests: one dict per chat with submit / first (first token) / done (last token) times
    and the number of tokens it generated."""
    ttfts = [r["first"] - r["submit"] for r in requests]
    latencies = [r["done"] - r["submit"] for r in requests]
    tpots = [(r["done"] - r["first"]) / (r["tokens"] - 1) for r in requests if r["tokens"] > 1]
    total_tokens = sum(r["tokens"] for r in requests)
    return {
        "n": n,
        "tokens": total_tokens,
        "wall": wall,
        "throughput": total_tokens / wall,
        "peak_active": peak_active,
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p99": _percentile(ttfts, 99),
        "lat_p50": _percentile(latencies, 50),
        "lat_p99": _percentile(latencies, 99),
        "tpot": statistics.mean(tpots) if tpots else 0.0,
        "kv_util": kv_util,
    }


def print_row(r):
    kv = f"{r['kv_util']:.1f}%" if r["kv_util"] is not None else "n/a"
    print(
        f"  {r['tokens']} tokens in {r['wall']:.2f}s -> {r['throughput']:.2f} tok/s | "
        f"TTFT p50 {r['ttft_p50']:.3f}s p99 {r['ttft_p99']:.3f}s | "
        f"latency p50 {r['lat_p50']:.3f}s p99 {r['lat_p99']:.3f}s | TPOT {r['tpot']:.3f}s | "
        f"peak active {r['peak_active']} | KV util {kv}"
    )


def write_row(engine_name, description, r):
    # the header and table columns live in RESULTS_FILE already, this only appends a row
    kv = f"{r['kv_util']:.1f}" if r["kv_util"] is not None else "–"
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"| {timestamp()} | {engine_name} | {description} | {device_label()} | {r['n']} | {REPLY_TOKENS} | {r['peak_active']} | {r['throughput']:.2f} | "
            f"{r['ttft_p50']:.3f} | {r['ttft_p99']:.3f} | {r['lat_p50']:.3f} | {r['lat_p99']:.3f} | "
            f"{r['tpot']:.3f} | {KV_RESERVED_MB:.0f} | {kv} |\n"
        )


# ---- our engine ----

_first_token_at = {}  # chat -> time its first reply token was sampled (end of prefill)
_last_token_at = {}  # chat -> time its reply finished generating


def _instrument_engine():
    """Timestamp the scheduler's own events so TTFT / latency mean the same thing as the HF
    run's per-token timestamps: first token = right after prefill samples it, last token = when
    the worker retires the job (before its bookkeeping-only trailing "\\n" forward pass)."""
    admit, finish = engine._admit, engine._finish

    def timed_admit(job):
        admit(job)
        _first_token_at[job.chat] = time.perf_counter()

    def timed_finish(job):
        _last_token_at[job.chat] = time.perf_counter()
        finish(job)

    engine._admit = timed_admit
    engine._finish = timed_finish
    engine.STOP_SEQUENCES = []  # replies always run the full REPLY_TOKENS, same as the HF run


def _run_chat(model, chat, text, prompt_len, trailing_len, result):
    ids_before = len(chat.ids)
    result["submit"] = time.perf_counter()
    for _ in chat.send_stream(model, text):
        pass
    result["first"] = _first_token_at.pop(chat)
    result["done"] = _last_token_at.pop(chat)
    # everything appended to chat.ids beyond the prompt and the trailing "\n" was generated
    result["tokens"] = len(chat.ids) - ids_before - prompt_len - trailing_len
    chat.release()  # give the cache blocks back so a deferred chat can be admitted


def run_level(model, tokenizer, n):
    """Start n chats at once, wait for every reply, and return that level's metrics."""
    trailing_len = len(tokenizer.encode("\n"))
    chats = []
    for _ in range(n):
        chat = Chat(tokenizer)
        chat.eos_id = NO_EOS
        chats.append(chat)
    texts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    prompt_lens = [len(tokenizer.encode(prompt_text(i))) for i in range(n)]
    results = [{} for _ in range(n)]

    # sample occupancy from Python-side state only (touching the cache tensor from this thread
    # would force a device sync in the middle of the worker's forward passes)
    peak = {"active": 0, "cached_tokens": 0}
    stop_sampling = threading.Event()

    def sampler():
        while not stop_sampling.is_set():
            peak["active"] = max(peak["active"], len(engine.active))
            cached = sum(len(c.ids) for c in chats if c.seq is not None)
            peak["cached_tokens"] = max(peak["cached_tokens"], cached)
            time.sleep(SAMPLE_INTERVAL)

    threads = [
        threading.Thread(target=_run_chat, args=(model, chats[i], texts[i], prompt_lens[i], trailing_len, results[i]))
        for i in range(n)
    ]
    sampler_thread = threading.Thread(target=sampler)
    sampler_thread.start()
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    stop_sampling.set()
    sampler_thread.join()

    kv_util = 100 * peak["cached_tokens"] * KV_BYTES_PER_TOKEN / 2**20 / KV_RESERVED_MB
    return summarize(n, results, wall, peak["active"], kv_util)


def warmup(model, tokenizer):
    engine.MAX_REPLY_TOKENS = 2  # the worker reads this module global on every step
    chat = Chat(tokenizer)
    chat.eos_id = NO_EOS
    for _ in chat.send_stream(model, PROMPTS[0]):
        pass
    chat.release()
    _first_token_at.clear()
    _last_token_at.clear()
    engine.MAX_REPLY_TOKENS = REPLY_TOKENS


def main():
    description = get_description()

    print("loading tokenizer and model...")
    tokenizer = QwenTokenizer(MODEL_PATH)
    model = load_qwen(MODEL_PATH)
    torch.manual_seed(SEED)
    _instrument_engine()

    print("warming up...")
    warmup(model, tokenizer)

    for n in CONCURRENCY_LEVELS:
        print(f"running {n} concurrent chat(s) on {device_label()}...")
        r = run_level(model, tokenizer, n)
        print_row(r)
        write_row("ours", description, r)  # append as each level finishes, so a partial sweep still lands

    print(f"results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
