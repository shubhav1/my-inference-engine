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


STOP_SEQUENCE = "\nUser:"  # model hallucinating the next turn instead of ending its own


def _overlap_len(text, pattern):
    """Length of the longest suffix of `text` that could still grow into `pattern`."""
    for k in range(min(len(text), len(pattern) - 1), 0, -1):
        if text.endswith(pattern[:k]):
            return k
    return 0


def generate(model, past_cache, new_ids, eos_id, max_tokens=MAX_REPLY_TOKENS):
    input_id = torch.tensor([new_ids], device=DEVICE)
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, pres_cache = model(input_id, full_past_kv = past_cache)
        past_cache = pres_cache
        next_id = logits[0, -1].argmax().item()
        yield next_id, pres_cache
        if next_id == eos_id:
            break
        input_id = torch.tensor([[next_id]], device=DEVICE)


class Chat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.cache = None
        self.messages = []
        self.eos_id = tokenizer.encoder["<|endoftext|>"]

    def send_stream(self, model, text):
        """Yield decoded reply pieces as they're generated, updating chat state as it goes."""

        self.messages.append(("user", text))
        new_ids = self.tokenizer.encode(f"User: {text}\nAssistant:")
        self.ids.extend(new_ids)
        reply_ids = []
        clean_reply = ""
        buffer = ""  # decoded text held back in case it's the start of STOP_SEQUENCE

        for token_id, cache in generate(model, self.cache, new_ids, self.eos_id):
            self.cache = cache
            self.ids.append(token_id)
            reply_ids.append(token_id)
            if token_id == self.eos_id:  # don't display the literal "<|endoftext|>"
                break

            buffer += self.tokenizer.decode([token_id])
            if STOP_SEQUENCE in buffer:  # model started hallucinating the next turn, kill it
                buffer = buffer.split(STOP_SEQUENCE)[0]
                if buffer:
                    clean_reply += buffer
                    yield buffer
                break

            overlap = _overlap_len(buffer, STOP_SEQUENCE)
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
        with torch.no_grad():
            _, self.cache = model(torch.tensor([trailing], device=DEVICE), full_past_kv=self.cache)
        self.messages.append(("assistant", clean_reply))

    def send(self, model, text):
        for piece in self.send_stream(model, text):
            print(piece, end="", flush=True)
        print()
        return self.messages[-1][1]

    def transcript(self):
        return self.tokenizer.decode(self.ids)


def main():
    print("loading tokenizer and model...")
    tokenizer = QwenTokenizer(MODEL_PATH)
    model = load_qwen(MODEL_PATH)
    print("ready. commands: /new <name>, /switch <name>, /list, /history, /quit")

    chats = {}
    current = None

    while True:
        try:
            line = input(f"[{current or 'no chat'}]> ").strip()
        except EOFError:
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        elif line == "/list":
            print(", ".join(chats) or "(no chats yet)")
        elif line.startswith("/new "):
            name = line[len("/new "):].strip()
            chats[name] = Chat(tokenizer)
            current = name
            print(f"started chat '{name}'")
        elif line.startswith("/switch "):
            name = line[len("/switch "):].strip()
            if name in chats:
                current = name
            else:
                print(f"no such chat: {name}")
        elif line == "/history":
            print(chats[current].transcript() if current else "no active chat")
        elif current is None:
            print("start a chat first with /new <name>")
        else:
            chats[current].send(model, line)


if __name__ == "__main__":
    main()
