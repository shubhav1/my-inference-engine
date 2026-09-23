import queue
import threading
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from kv_cache import KVCache
from model import Qwen2ForCausalLM
from tokenizer import QwenTokenizer

MODEL_PATH = "./qwen2.5-1.5b"
MAX_REPLY_TOKENS = 512  # safety net against a reply that never hits EOS
TEMPERATURE = 1

# Sizes the shared static KV cache (see kv_cache.py): MAX_BATCH_SIZE is how many chats can
# hold a live cache slot at once (a slot is held for a chat's whole lifetime, not just while
# it's actively streaming, since cache must persist across turns), the (MAX_BATCH_SIZE + 1)th
# chat to send a first message simply waits for one to free up (see `_deferred` in _worker).
# MAX_SEQ_LEN caps total tokens (prompt + every turn + every reply) any one chat can reach.
MAX_BATCH_SIZE = 8
MAX_SEQ_LEN = 2048

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

KV_CACHE = KVCache(MAX_BATCH_SIZE, MAX_SEQ_LEN, DEVICE, dtype=torch.float32)


def load_qwen(model_path):
    model = Qwen2ForCausalLM()
    state_dict = load_file(f"{model_path}/model.safetensors")
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


STOP_SEQUENCES = ["\nUser:", "\nUser"]  # model hallucinating the next turn instead of ending its own


def _overlap_len(text, patterns):
    """Length of the longest suffix of `text` that could still grow into any of `patterns`."""
    best = 0
    for pattern in patterns:
        for k in range(min(len(text), len(pattern) - 1), 0, -1):
            if text.endswith(pattern[:k]):
                best = max(best, k)
                break
    return best


def _find_stop(text, patterns):
    """The earliest-starting stop pattern found in `text`, or None."""
    best = None
    for pattern in patterns:
        i = text.find(pattern)
        if i != -1 and (best is None or i < text.find(best)):
            best = pattern
    return best


def generate(model, new_ids, eos_id, max_tokens=MAX_REPLY_TOKENS):
    """Single-sequence reference generation loop (used by the benchmark harness): allocates
    its own private single-slot KV cache, prefills the prompt, then decodes up to max_tokens."""
    cache = KVCache(1, MAX_SEQ_LEN, DEVICE, dtype=next(model.parameters()).dtype)
    slot = cache.alloc()

    next_id = new_ids
    is_first = True
    for _ in range(max_tokens):
        with torch.no_grad():
            input_id = torch.tensor([next_id] if is_first else [[next_id]], device=DEVICE)
            is_first = False
            logits = model(input_id, cache, [slot])
        next_id = _sample(logits[0, -1])
        yield next_id
        if next_id == eos_id:
            break


class _Job:
    def __init__(self, chat, model, text, out_queue):
        self.chat = chat
        self.model = model
        self.text = text
        self.out_queue = out_queue
        self.buffer = ""  # decoded text held back in case it's the start of a stop sequence
        self.steps = 0
        self.next_token = None  # token most recently produced, to be fed on this job's next decode call


def _sample(logit_row):
    probs = F.softmax(logit_row / TEMPERATURE, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


_job_queue = queue.Queue()
active = []
_deferred = []  # jobs waiting on a new chat's first message, held back until a cache slot frees


def _needs_new_slot(job):
    return job.chat.slot is None


def _admit(job):
    """Prefill: one forward pass over the whole prompt, explicitly separate from decoding.

    Runs once per job, right before it joins `active` and starts sharing decode steps.
    """
    chat, model, text = job.chat, job.model, job.text

    chat.pending.remove(text)  # this message is now being handled, not just waiting
    chat.messages.append(("user", text))
    new_ids = chat.tokenizer.encode(f"User: {text}\nAssistant:")
    chat.ids.extend(new_ids)
    chat.listeners = [job.out_queue]  # set before in_progress: closes the window watch() could miss it
    chat.in_progress = ""  # visible to any page render while this reply is still streaming

    if chat.slot is None:  # first message of this chat's lifetime, reserve its cache row
        chat.slot = KV_CACHE.alloc()

    input_ids = torch.tensor([new_ids], device=DEVICE)
    with torch.no_grad():
        logits = model(input_ids, KV_CACHE, [chat.slot])
    job.next_token = _sample(logits[0, -1])

    active.append(job)


def _emit(job, piece):
    job.chat.in_progress += piece
    for q in job.chat.listeners:
        q.put(piece)


def _finish(job):
    chat = job.chat
    trailing = chat.tokenizer.encode("\n")
    chat.ids.extend(trailing)
    with torch.no_grad():
        job.model(torch.tensor([trailing], device=DEVICE), KV_CACHE, [chat.slot])
    # chat.slot is NOT freed here -- the conversation continues across turns, so its cache
    # row has to survive until the chat itself is cleared or deleted (see Chat.release).
    chat.messages.append(("assistant", chat.in_progress))
    chat.in_progress = None
    for q in chat.listeners:
        q.put(None)
    chat.listeners = []
    active.remove(job)


def _try_admit_or_defer(job):
    """Admit `job` if it can get a cache slot; otherwise hold it until one frees up.

    A follow-up message in a chat that already owns a slot always proceeds, only a chat's
    very first message needs a fresh one, and the static cache has a fixed number of rows.
    """
    if not _needs_new_slot(job) or KV_CACHE.free_slot_count() > 0:
        _admit(job)
    else:
        _deferred.append(job)


def _worker():
    while True:
        # a slot may have freed up (a chat got cleared/deleted) since we last checked -- re-check
        # per job, since admitting one can take the last free slot another was counting on
        for job in list(_deferred):
            if not _needs_new_slot(job) or KV_CACHE.free_slot_count() > 0:
                _deferred.remove(job)
                _admit(job)

        if not active and not _deferred:
            _try_admit_or_defer(_job_queue.get())  # block until at least one request comes in
        elif not active and _deferred:
            # nothing decoding and the deferred job(s) still have no free slot -- avoid
            # busy-spinning on an empty `active` list while waiting for one to free up
            time.sleep(0.05)
        elif not _job_queue.empty():
            # admit at most one new job per tick: _admit() runs a full, un-chunked prefill
            # over that job's prompt, which blocks every already-active job's decode step
            # for as long as it takes. Draining the whole queue here would stack those
            # prefills back-to-back before any already-streaming reply got to advance again;
            # admitting one at a time interleaves a decode step in between instead.
            _try_admit_or_defer(_job_queue.get())
        _process(active)


def _process(active):
    """One decode step for every job currently active, in FIFO order."""

    # for loop for bookkeeping stuff
    continuing = []
    for job in list(active):  # snapshot, _finish() removes from `active` as jobs complete
        chat = job.chat
        token_id = job.next_token
        chat.ids.append(token_id)

        if token_id == chat.eos_id:  # don't display "<|endoftext|>"
            _finish(job)
            continue

        job.buffer += chat.tokenizer.decode([token_id])
        stop = _find_stop(job.buffer, STOP_SEQUENCES)  # model hallucinating the next turn, kill it
        if stop is not None:
            job.buffer = job.buffer.split(stop)[0]
            if job.buffer:
                _emit(job, job.buffer)
            _finish(job)
            continue

        overlap = _overlap_len(job.buffer, STOP_SEQUENCES)
        release, job.buffer = job.buffer[:len(job.buffer) - overlap], job.buffer[len(job.buffer) - overlap:]
        if release:
            _emit(job, release)

        job.steps += 1
        if job.steps >= MAX_REPLY_TOKENS:  # safety net against a reply that never hits EOS
            if job.buffer:
                _emit(job, job.buffer)
            _finish(job)
            continue

        continuing.append(job)

    if not continuing:
        return

    # run every continuing job's next decode step together, in one batched forward pass --
    # each job just points at its own row (slot) in the shared static KV cache, so there's
    # no padding or copying to assemble here regardless of how ragged their lengths are
    model = continuing[0].model
    slots = [job.chat.slot for job in continuing]
    input_ids = torch.tensor([[job.next_token] for job in continuing], device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids, KV_CACHE, slots)

    for i, job in enumerate(continuing):
        job.next_token = _sample(logits[i, -1])


threading.Thread(target=_worker, daemon=True).start()


class Chat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.slot = None  # this chat's row in KV_CACHE, reserved on its first message
        self.messages = []
        self.pending = []  # user texts submitted but not yet picked up by the worker
        self.in_progress = None  # reply text generated so far, while one is actively streaming
        self.listeners = []  # queues currently watching the active generation, if any
        self.eos_id = tokenizer.encoder["<|endoftext|>"]

    def release(self):
        """Free this chat's KV-cache slot, if it holds one. Call when the chat is cleared or
        deleted so another chat can use that row, the cache has a fixed number of them."""
        if self.slot is not None:
            KV_CACHE.free(self.slot)
            self.slot = None

    def send_stream(self, model, text):
        """Yield decoded reply pieces as they're generated, updating chat state as it goes.

        Every message is enqueued and handled by a single background worker: a new request is
        prefilled once (`_admit`), then joins `active` and gets one decode step per worker loop
        alongside every other in-flight reply, so multiple chats generate concurrently without
        ever touching the model from more than one thread. The message itself is recorded
        immediately (`pending`), though, so it's never invisible while queued behind others.
        """
        self.pending.append(text)
        out_queue = queue.Queue()
        _job_queue.put(_Job(self, model, text, out_queue))
        while True:
            piece = out_queue.get()
            if piece is None:
                break
            yield piece

    def all_messages(self):
        """Committed history, the reply streaming in right now (if any), and anything still queued."""
        result = list(self.messages)
        if self.in_progress is not None:
            result.append(("assistant", self.in_progress))
        result.extend(("user", text) for text in self.pending)
        return result

    def watch(self):
        """Join the reply currently streaming, if any.

        Returns (text_so_far, queue) to catch up on and then read live pieces from, or
        (None, None) if nothing is actively generating right now.
        """
        if self.in_progress is None:
            return None, None
        q = queue.Queue()
        self.listeners.append(q)
        return self.in_progress, q

    def send(self, model, text):
        for piece in self.send_stream(model, text):
            print(piece, end="", flush=True)
        print()
        return self.messages[-1][1]

    def transcript(self):
        return self.tokenizer.decode(self.ids)
