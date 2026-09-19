import queue
import threading

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from model import NUM_LAYERS, Qwen2ForCausalLM
from tokenizer import QwenTokenizer

MODEL_PATH = "./qwen2.5-1.5b"
MAX_REPLY_TOKENS = 512  # safety net against a reply that never hits EOS
TEMPERATURE = 1

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


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


def generate(model, past_cache, new_ids, eos_id, max_tokens=MAX_REPLY_TOKENS):
    next_id = new_ids
    is_first = True
    for _ in range(max_tokens):
        with torch.no_grad():
            input_id = torch.tensor([next_id] if is_first else [[next_id]], device=DEVICE)
            is_first = False
            logits, pres_cache = model(input_id, full_past_kv=past_cache)
        past_cache = pres_cache
        # next_id = logits[0, -1].argmax().item()
        probs = F.softmax(logits[0, -1] / TEMPERATURE, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        yield next_id, pres_cache
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

    input_ids = torch.tensor([new_ids], device=DEVICE)
    with torch.no_grad():
        logits, cache = model(input_ids, full_past_kv=chat.cache)
    chat.cache = cache
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
        _, chat.cache = job.model(torch.tensor([trailing], device=DEVICE), full_past_kv=chat.cache)
    chat.messages.append(("assistant", chat.in_progress))
    chat.in_progress = None
    for q in chat.listeners:
        q.put(None)
    chat.listeners = []
    active.remove(job)


def _worker():
    while True:
        if not active:
            _admit(_job_queue.get())  # block until at least one request comes in
        elif not _job_queue.empty():
            # admit at most one new job per tick: _admit() runs a full, un-chunked prefill
            # over that job's prompt, which blocks every already-active job's decode step
            # for as long as it takes. Draining the whole queue here would stack those
            # prefills back-to-back before any already-streaming reply got to advance again;
            # admitting one at a time interleaves a decode step in between instead.
            _admit(_job_queue.get())
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

    # run every continuing job's next decode step together, in one batched forward pass
    model = continuing[0].model
    past_lens = [job.chat.cache[0][0].shape[2] for job in continuing]
    max_len = max(past_lens)

    batched_cache = []
    for layer in range(NUM_LAYERS):
        ks, vs = [], []
        for job in continuing:
            k, v = job.chat.cache[layer]
            pad = max_len - k.shape[2]
            if pad:  # left-pad shorter caches with zeros so every row lines up at the same length
                zeros_k = torch.zeros((1, k.shape[1], pad, k.shape[3]), dtype=k.dtype, device=k.device)
                zeros_v = torch.zeros((1, v.shape[1], pad, v.shape[3]), dtype=v.dtype, device=v.device)
                k = torch.cat([zeros_k, k], dim=2)
                v = torch.cat([zeros_v, v], dim=2)
            ks.append(k)
            vs.append(v)
        batched_cache.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

    input_ids = torch.tensor([[job.next_token] for job in continuing], device=DEVICE)
    position_ids = torch.tensor([[p] for p in past_lens], device=DEVICE)  # each row's unpadded position

    if max_len == min(past_lens):
        # every row is already the same length (the common case -- one job alone, or several
        # jobs that happen to line up) -- nothing is actually padded, so skip building a mask
        # and let attention use its fused causal kernel instead
        pad_mask = None
    else:
        pad_mask = torch.zeros(len(continuing), max_len + 1, dtype=torch.bool, device=DEVICE)
        for i, p in enumerate(past_lens):
            pad_mask[i, max_len - p:] = True  # real past (right-aligned) plus this step's new token

    with torch.no_grad():
        logits, present_kv = model(
            input_ids, full_past_kv=batched_cache, position_ids=position_ids, pad_mask=pad_mask
        )

    for i, job in enumerate(continuing):
        new_len = past_lens[i] + 1
        # slice this row's own (unpadded) cache back out -- padding only existed for the shared call
        job.chat.cache = [(k[i:i + 1, :, -new_len:, :], v[i:i + 1, :, -new_len:, :]) for k, v in present_kv]
        job.next_token = _sample(logits[i, -1])


threading.Thread(target=_worker, daemon=True).start()


class Chat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.cache = None
        self.messages = []
        self.pending = []  # user texts submitted but not yet picked up by the worker
        self.in_progress = None  # reply text generated so far, while one is actively streaming
        self.listeners = []  # queues currently watching the active generation, if any
        self.eos_id = tokenizer.encoder["<|endoftext|>"]

    def send_stream(self, model, text):
        """Yield decoded reply pieces as they're generated, updating chat state as it goes.

        Every message is enqueued and handled by a single background worker: a new request is
        prefilled once (`_admit`), then joins `active` and gets one decode step per worker loop
        alongside every other in-flight reply -- so multiple chats generate concurrently without
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
