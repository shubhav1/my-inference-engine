import torch
from safetensors.torch import load_file
from model import Qwen2ForCausalLM
from tokenizer import QwenTokenizer

MODEL_PATH = "./qwen2.5-1.5b"
MAX_REPLY_TOKENS = 512  # safety net against a reply that never hits EOS
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_qwen(model_path):
    model = Qwen2ForCausalLM()
    state_dict = load_file(f"{model_path}/model.safetensors")
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


def generate(model, ids, eos_id, max_tokens=MAX_REPLY_TOKENS):
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=DEVICE))
        next_id = logits[0, -1].argmax().item()
        ids.append(next_id)
        yield next_id
        if next_id == eos_id:
            break


class Chat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.messages = []
        self.eos_id = tokenizer.encoder["<|endoftext|>"]

    def send_stream(self, model, text):
        """Yield decoded reply pieces as they're generated, updating chat state as it goes."""
        self.messages.append(("user", text))
        self.ids.extend(self.tokenizer.encode(f"User: {text}\nAssistant:"))
        reply_ids = []
        for token_id in generate(model, self.ids, self.eos_id):
            reply_ids.append(token_id)
            yield self.tokenizer.decode([token_id])
        self.ids.extend(self.tokenizer.encode("\n"))
        self.messages.append(("assistant", self.tokenizer.decode(reply_ids)))

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
