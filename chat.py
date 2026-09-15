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

# MPS isn't safe for concurrent forward passes from multiple threads at once -- the web app can
# have several chats generating at the same time, so every call into the model is serialized.
_model_lock = threading.Lock()


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
        with torch.no_grad(), _model_lock:
            input_id = torch.tensor([next_id] if is_first else [[next_id]], device=DEVICE)
            is_first = False
            logits, pres_cache = model(input_id, full_past_kv=past_cache)
        past_cache = pres_cache
        next_id = logits[0, -1].argmax().item()
        yield next_id, pres_cache
        if next_id == eos_id:
            break


class Chat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.cache = None
        self.messages = []
        self.eos_id = tokenizer.encoder["<|endoftext|>"]
        self.turn_lock = threading.Lock()  # one turn at a time; a second message waits for the first

    def send_stream(self, model, text):
        """Yield decoded reply pieces as they're generated, updating chat state as it goes."""
        with self.turn_lock:
            self.messages.append(("user", text))
            new_ids = self.tokenizer.encode(f"User: {text}\nAssistant:")
            self.ids.extend(new_ids)
            clean_reply = ""
            buffer = ""  # decoded text held back in case it's the start of a stop sequence

            for token_id, cache in generate(model, self.cache, new_ids, self.eos_id):
                self.cache = cache
                self.ids.append(token_id)
                if token_id == self.eos_id:  # don't display the literal "<|endoftext|>"
                    break

                buffer += self.tokenizer.decode([token_id])
                stop = _find_stop(buffer, STOP_SEQUENCES)  # model hallucinating the next turn, kill it
                if stop is not None:
                    buffer = buffer.split(stop)[0]
                    if buffer:
                        clean_reply += buffer
                        yield buffer
                    break

                overlap = _overlap_len(buffer, STOP_SEQUENCES)
                release, buffer = buffer[:len(buffer) - overlap], buffer[len(buffer) - overlap:]
                if release:
                    clean_reply += release
                    yield release
            else:
                if buffer:  # loop hit max_tokens with a held-back tail that was never a real match
                    clean_reply += buffer
                    yield buffer

            trailing = self.tokenizer.encode("\n")
            self.ids.extend(trailing)
            with torch.no_grad(), _model_lock:
                _, self.cache = model(torch.tensor([trailing], device=DEVICE), full_past_kv=self.cache)
            self.messages.append(("assistant", clean_reply))

    def send(self, model, text):
        for piece in self.send_stream(model, text):
            print(piece, end="", flush=True)
        print()
        return self.messages[-1][1]

    def transcript(self):
        return self.tokenizer.decode(self.ids)