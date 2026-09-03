from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).cuda()
prompt = "Explain latency in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50
    )
print(tokenizer.decode(output[0]))