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

# Sizes the shared paged KV cache (see kv_cache.py): KV_CACHE_TOKENS is the total token
# budget every chat draws from, carved into BLOCK_SIZE-token blocks that a chat takes only as
# it grows. There is no fixed number of chats: a message is admitted whenever there are enough
# free blocks for its prompt plus a full reply, evicting idle chats' caches if needed (they
# re-prefill from their token history on their next turn), otherwise it waits in `_deferred`.
# MAX_SEQ_LEN only sizes generate()'s private single-sequence cache.
BLOCK_SIZE = 16
KV_CACHE_TOKENS = 8 * 2048  # same memory budget the old static cache had (8 slots x 2048 tokens)
MAX_SEQ_LEN = 2048

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# bf16 on CUDA (what the FA2 kernel is tuned for), fp32 elsewhere to keep mps results comparable
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

KV_CACHE = KVCache(KV_CACHE_TOKENS // BLOCK_SIZE, BLOCK_SIZE, DEVICE, dtype=DTYPE)


def load_qwen(model_path):
    model = Qwen2ForCausalLM()
    state_dict = load_file(f"{model_path}/model.safetensors")
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE, dtype=DTYPE)


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
    its own private KV cache, prefills the prompt, then decodes up to max_tokens."""
    cache = KVCache(MAX_SEQ_LEN // BLOCK_SIZE, BLOCK_SIZE, DEVICE, dtype=next(model.parameters()).dtype)
    seq = cache.alloc()

    next_id = new_ids
    is_first = True
    for _ in range(max_tokens):
        with torch.no_grad():
            input_id = torch.tensor([next_id] if is_first else [[next_id]], device=DEVICE)
            is_first = False
            logits = model(input_id, cache, [seq])
        next_id = _sample(logits[:, -1])[0]
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
        self.new_ids = chat.tokenizer.encode(f"User: {text}\nAssistant:")


def _sample(logits):
    """One token id per row of [B, vocab] logits, each row sampled independently. Batched so a
    decode step pays a single device sync for every active chat, not one per chat."""
    probs = F.softmax(logits / TEMPERATURE, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1).tolist()


_job_queue = queue.Queue()
active = []
_deferred = []  # jobs waiting for enough free KV-cache blocks to be admitted
_resident = []  # chats whose history is in KV_CACHE, least recently used first
_cache_lock = threading.Lock()  # Chat.release() runs on request threads, eviction on the worker


def _reserve_len(job):
    """Tokens job's chat can reach by the end of this turn: its history, this prompt, a full
    reply, and the "\n" that _finish() appends -- reserved up front so decode never runs dry."""
    trailing = len(job.chat.tokenizer.encode("\n"))
    return len(job.chat.ids) + len(job.new_ids) + MAX_REPLY_TOKENS + trailing


def _blocks_needed(job):
    return KV_CACHE.blocks_needed(job.chat.seq, _reserve_len(job))


def _evict(chat):
    """Drop an idle chat's cache. Its token history stays in chat.ids, so its next turn simply
    re-prefills it (see _admit) -- trading recompute for not pinning memory between turns."""
    KV_CACHE.free(chat.seq)
    chat.seq = None
    _resident.remove(chat)


def _make_room(job):
    """Free blocks until `job` fits, evicting idle chats' caches oldest-first. False if it still
    can't fit because the rest is held by replies that are mid-generation."""
    busy = {j.chat for j in active} | {job.chat}
    with _cache_lock:
        while _blocks_needed(job) > KV_CACHE.free_block_count():
            idle = [c for c in _resident if c not in busy]
            if not idle:
                return False
            _evict(idle[0])
    return True


def _touch(chat):
    """Mark chat most recently used, so eviction picks it last."""
    with _cache_lock:
        if chat in _resident:
            _resident.remove(chat)
        _resident.append(chat)


def _admit(job):
    """Prefill: one forward pass over the whole prompt, explicitly separate from decoding.

    Runs once per job, right before it joins `active` and starts sharing decode steps.
    """
    chat, model, text = job.chat, job.model, job.text

    reserve_len = _reserve_len(job)  # before chat.ids grows below
    chat.pending.remove(text)  # this message is now being handled, not just waiting
    chat.messages.append(("user", text))
    chat.ids.extend(job.new_ids)
    chat.listeners = [job.out_queue]  # set before in_progress: closes the window watch() could miss it
    chat.in_progress = ""  # visible to any page render while this reply is still streaming

    if chat.seq is None:
        # first message, or this chat's cache was evicted while idle: prefill its whole history
        chat.seq = KV_CACHE.alloc()
    # everything in chat.ids the cache doesn't hold yet: this message, plus the previous reply's
    # last token and trailing "\n" that _finish() leaves for this prefill to write
    prefill_ids = chat.ids[KV_CACHE.length(chat.seq):]
    KV_CACHE.ensure_capacity(chat.seq, reserve_len)  # _make_room() already checked these are free
    _touch(chat)

    input_ids = torch.tensor([prefill_ids], device=DEVICE)
    with torch.no_grad():
        logits = model(input_ids, KV_CACHE, [chat.seq])
    job.next_token = _sample(logits[:, -1])[0]

    active.append(job)


def _emit(job, piece):
    job.chat.in_progress += piece
    for q in job.chat.listeners:
        q.put(piece)


def _finish(job):
    chat = job.chat
    # the reply's last token was appended to chat.ids but never fed back through the model. It
    # and the trailing "\n" are left out of the cache on purpose: the chat's next prefill writes
    # them along with its new message (see _admit), instead of a separate forward pass per reply
    # that would stall every other active chat
    chat.ids.extend(chat.tokenizer.encode("\n"))
    # the cache is NOT freed here -- the conversation continues across turns -- but blocks
    # reserved for reply tokens that were never generated go back to the pool
    with _cache_lock:
        KV_CACHE.trim(chat.seq)
    _touch(chat)
    chat.messages.append(("assistant", chat.in_progress))
    chat.in_progress = None
    for q in chat.listeners:
        q.put(None)
    chat.listeners = []
    active.remove(job)


def _reject(job, reason):
    """Answer a message that can never fit in the cache, instead of deferring it forever."""
    job.chat.pending.remove(job.text)
    job.out_queue.put(f"[error: {reason}]")
    job.out_queue.put(None)


def _try_admit_or_defer(job):
    """Admit `job` if its prompt plus a full reply fits in free blocks (evicting idle chats if
    needed); otherwise hold it until replies in flight finish and give blocks back."""
    if KV_CACHE.blocks_needed(None, _reserve_len(job)) > KV_CACHE.num_blocks:
        _reject(job, "conversation is too long for the KV cache")
    elif _make_room(job):
        _admit(job)
    else:
        _deferred.append(job)


def _worker():
    while True:
        # blocks may have freed up (a reply finished, a chat got cleared) since we last checked --
        # re-check per job, since admitting one can take the blocks another was counting on
        for job in list(_deferred):
            if _make_room(job):
                _deferred.remove(job)
                _admit(job)

        if not active and not _deferred:
            _try_admit_or_defer(_job_queue.get())  # block until at least one request comes in
        elif not active and _deferred:
            # nothing decoding and the deferred job(s) still don't fit -- avoid busy-spinning
            # on an empty `active` list while waiting for a chat to be released
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
    # each job just points at its own block table in the shared paged KV cache, so there's
    # no padding or copying to assemble here regardless of how ragged their lengths are
    model = continuing[0].model
    seqs = [job.chat.seq for job in continuing]
    input_ids = torch.tensor([[job.next_token] for job in continuing], device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids, KV_CACHE, seqs)

    for job, token_id in zip(continuing, _sample(logits[:, -1])):
        job.next_token = token_id


threading.Thread(target=_worker, daemon=True).start()


class Chat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.seq = None  # this chat's sequence in KV_CACHE, None until first message or after eviction
        self.messages = []
        self.pending = []  # user texts submitted but not yet picked up by the worker
        self.in_progress = None  # reply text generated so far, while one is actively streaming
        self.listeners = []  # queues currently watching the active generation, if any
        self.eos_id = tokenizer.encoder["<|endoftext|>"]

    def release(self):
        """Free this chat's KV-cache blocks, if it holds any. Call when the chat is cleared or
        deleted so other chats can use them right away instead of waiting for eviction."""
        with _cache_lock:
            if self.seq is not None:
                _evict(self)

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
