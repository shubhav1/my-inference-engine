import torch
from safetensors.torch import load_file
from model import Qwen2ForCausalLM
from tokenizer import QwenTokenizer

MODEL_PATH = "./qwen2.5-1.5b"

def load_qwen(model_path):
    model = Qwen2ForCausalLM()
    state_dict = load_file(f"{model_path}/model.safetensors")
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# run model
tokenizer = QwenTokenizer(MODEL_PATH)
model = load_qwen(MODEL_PATH)

text = input("Enter text: ")
input_ids = torch.tensor([tokenizer.encode(text)])

for _ in range(10):
    with torch.no_grad():
        logits = model(input_ids)

    next_token_id = logits[0, -1].argmax().item()
    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

    print(tokenizer.decode([next_token_id]), end="")
