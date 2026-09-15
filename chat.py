import queue
import threading

import torch
from safetensors.torch import load_file
from model import Qwen2ForCausalLM
from tokenizer import QwenTokenizer

MODEL_PATH = "./qwen2.5-1.5b"
MAX_REPLY_TOKENS = 512  # safety net against a reply that never hits EOS
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
        next_id = logits[0, -1].argmax().item()
        yield next_id, pres_cache
        if next_id == eos_id:
            break


class _Job:
    def __init__(self, chat, model, text, out_queue):
        self.chat = chat
        self.model = model
        self.text = text
        self.out_queue = out_queue


_job_queue = queue.Queue()


def _worker():
    while True:
        job = _job_queue.get()
        _process(job)


def _process(job):
    chat, model, text, out_queue = job.chat, job.model, job.text, job.out_queue

    chat.pending.remove(text)  # this message is now being handled, not just waiting
    chat.messages.append(("user", text))
    new_ids = chat.tokenizer.encode(f"User: {text}\nAssistant:")
    chat.ids.extend(new_ids)
    chat.listeners = [out_queue]  # set before in_progress: closes the window where watch() could miss it
    chat.in_progress = ""  # visible to any page render while this reply is still streaming
    buffer = ""  # decoded text held back in case it's the start of a stop sequence

    def emit(piece):
        chat.in_progress += piece
        for q in chat.listeners:
            q.put(piece)

    for token_id, cache in generate(model, chat.cache, new_ids, chat.eos_id):
        chat.cache = cache
        chat.ids.append(token_id)
        if token_id == chat.eos_id:  # don't display "<|endoftext|>"
            break

        buffer += chat.tokenizer.decode([token_id])
        stop = _find_stop(buffer, STOP_SEQUENCES)  # model hallucinating the next turn, kill it
        if stop is not None:
            buffer = buffer.split(stop)[0]
            if buffer:
                emit(buffer)
            break

        overlap = _overlap_len(buffer, STOP_SEQUENCES)
        release, buffer = buffer[:len(buffer) - overlap], buffer[len(buffer) - overlap:]
        if release:
            emit(release)
    else:
        if buffer:  # loop hit max_tokens with a held-back tail that was never a real match
            emit(buffer)

    trailing = chat.tokenizer.encode("\n")
    chat.ids.extend(trailing)
    with torch.no_grad():
        _, chat.cache = model(torch.tensor([trailing], device=DEVICE), full_past_kv=chat.cache)
    chat.messages.append(("assistant", chat.in_progress))
    chat.in_progress = None
    for q in chat.listeners:
        q.put(None)
    chat.listeners = []


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

        Every message, from every chat, is enqueued and handled by a single background worker
        that processes one at a time in FIFO order -- this is what keeps two messages from ever
        touching the model or a chat's cache at the same time. The message itself is recorded
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
