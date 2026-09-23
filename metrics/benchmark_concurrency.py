import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # chat.py/tokenizer.py live one level up

import chat as engine
from chat import DEVICE, KV_CACHE, MAX_BATCH_SIZE, MODEL_PATH, Chat, load_qwen
from tokenizer import QwenTokenizer

# Drives the real scheduler in chat.py (admission, deferral, batched decode) with N chats that
# all send their first message at the same moment, then measures aggregate throughput, latency
# and how much of the KV cache is actually in use. Sweeping N past MAX_BATCH_SIZE shows where
# chats start queuing for cache capacity rather than running concurrently.
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
REPLY_TOKENS = 64  # every reply runs exactly this many tokens (EOS disabled) unless a stop sequence hits
NO_EOS = -1  # no real token id is negative, so a reply never ends early on EOS
SEED = 0
SAMPLE_INTERVAL = 0.01  # seconds between samples of scheduler / cache occupancy
RESULTS_FILE = Path(__file__).resolve().parent / "concurrency_results.md"

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


class _ChatRun:
    def __init__(self, chat, text):
        self.chat = chat
        self.text = text
        self.sent = None
        self.first_piece = None
        self.done = None
        self.tokens = 0


def _run_chat(run, model, prompt_len, trailing_len):
    ids_before = len(run.chat.ids)
    run.sent = time.perf_counter()
    for _ in run.chat.send_stream(model, run.text):
        if run.first_piece is None:
            run.first_piece = time.perf_counter()
    run.done = time.perf_counter()
    # everything appended to chat.ids beyond the prompt and the trailing "\n" was generated
    run.tokens = len(run.chat.ids) - ids_before - prompt_len - trailing_len
    run.chat.release()  # give the cache slot back so a deferred chat can be admitted


def _percentile(values, pct):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def run_level(model, tokenizer, n):
    """Start n chats at once, wait for every reply, and return that level's metrics."""
    trailing_len = len(tokenizer.encode("\n"))
    runs = []
    for i in range(n):
        chat = Chat(tokenizer)
        chat.eos_id = NO_EOS
        runs.append(_ChatRun(chat, PROMPTS[i % len(PROMPTS)]))
    prompt_lens = [len(tokenizer.encode(f"User: {r.text}\nAssistant:")) for r in runs]

    # sample occupancy from Python-side state only (touching the cache tensor from this thread
    # would force a device sync in the middle of the worker's forward passes)
    peak = {"active": 0, "deferred": 0, "slots": 0, "cached_tokens": 0}
    stop_sampling = threading.Event()

    def sampler():
        while not stop_sampling.is_set():
            holding = [r.chat for r in runs if r.chat.slot is not None]
            peak["active"] = max(peak["active"], len(engine.active))
            peak["deferred"] = max(peak["deferred"], len(engine._deferred))
            peak["slots"] = max(peak["slots"], len(holding))
            peak["cached_tokens"] = max(peak["cached_tokens"], sum(len(c.ids) for c in holding))
            time.sleep(SAMPLE_INTERVAL)

    threads = [
        threading.Thread(target=_run_chat, args=(r, model, plen, trailing_len))
        for r, plen in zip(runs, prompt_lens)
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

    total_tokens = sum(r.tokens for r in runs)
    ttfts = [r.first_piece - r.sent for r in runs if r.first_piece is not None]
    tpots = [
        (r.done - r.first_piece) / (r.tokens - 1)
        for r in runs if r.first_piece is not None and r.tokens > 1
    ]

    bytes_per_token = KV_CACHE.buf.numel() * KV_CACHE.buf.element_size() / (KV_CACHE.max_batch * KV_CACHE.max_len)
    reserved_mb = KV_CACHE.buf.numel() * KV_CACHE.buf.element_size() / 2**20
    used_mb = peak["cached_tokens"] * bytes_per_token / 2**20

    return {
        "n": n,
        "prompt_min": min(prompt_lens),
        "prompt_max": max(prompt_lens),
        "peak_active": peak["active"],
        "peak_deferred": peak["deferred"],
        "peak_slots": peak["slots"],
        "wall": wall,
        "tokens": total_tokens,
        "throughput": total_tokens / wall,
        "ttft_p50": statistics.median(ttfts),
        "ttft_p95": _percentile(ttfts, 95),
        "tpot": statistics.mean(tpots) if tpots else 0.0,
        "kv_reserved_mb": reserved_mb,
        "kv_used_mb": used_mb,
        "kv_util": 100 * used_mb / reserved_mb,
    }


def warmup(model, tokenizer):
    engine.MAX_REPLY_TOKENS = 2
    chat = Chat(tokenizer)
    chat.eos_id = NO_EOS
    for _ in chat.send_stream(model, PROMPTS[0]):
        pass
    chat.release()


def write_row(description, r):
    # the header and table columns live in RESULTS_FILE already, this only appends a row
    description = description.replace("|", "/").replace("\n", " ")
    now = datetime.now()

    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"| {now.strftime('%Y-%m-%d')} | {now.strftime('%H:%M:%S')} | {description} | "
            f"{DEVICE.type} | {MAX_BATCH_SIZE} | {r['n']} | {r['prompt_min']}-{r['prompt_max']} | "
            f"{REPLY_TOKENS} | {r['peak_active']} | {r['peak_deferred']} | {r['wall']:.2f} | "
            f"{r['throughput']:.2f} | {r['ttft_p50']:.3f} | {r['ttft_p95']:.3f} | {r['tpot']:.3f} | "
            f"{r['kv_reserved_mb']:.0f} | {r['kv_used_mb']:.0f} | {r['kv_util']:.1f} |\n"
        )


def main():
    description = input("Describe this run: ").strip()

    print("loading tokenizer and model...")
    tokenizer = QwenTokenizer(MODEL_PATH)
    model = load_qwen(MODEL_PATH)
    torch.manual_seed(SEED)

    print("warming up...")
    warmup(model, tokenizer)
    engine.MAX_REPLY_TOKENS = REPLY_TOKENS  # the worker reads this module global on every step

    for n in CONCURRENCY_LEVELS:
        print(f"running {n} concurrent chat(s) on {DEVICE.type}...")
        r = run_level(model, tokenizer, n)
        print(
            f"  {r['tokens']} tokens in {r['wall']:.2f}s -> {r['throughput']:.2f} tok/s | "
            f"TTFT p50 {r['ttft_p50']:.3f}s p95 {r['ttft_p95']:.3f}s | TPOT {r['tpot']:.3f}s | "
            f"peak active {r['peak_active']}, deferred {r['peak_deferred']} | "
            f"KV used {r['kv_used_mb']:.0f}/{r['kv_reserved_mb']:.0f} MB ({r['kv_util']:.1f}%)"
        )
        write_row(description, r)  # append as each level finishes, so a partial sweep still lands

    print(f"results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
