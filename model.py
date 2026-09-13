# implementation of qwen architecture
# Architecture: transformers with RoPE, SwiGLU, RMSNorm, Attention QKV bias and tied word embeddings
# Number of Layers: 28
# Number of Attention Heads (GQA): 12 for Q and 2 for KV

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen2.5-1.5B architecture constants (from config.json)
HIDDEN_SIZE = 1536
NUM_LAYERS = 28
NUM_HEADS = 12
NUM_KV_HEADS = 2
N_REP = NUM_HEADS // NUM_KV_HEADS
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
INTERMEDIATE_SIZE = 8960
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1_000_000.0
""" config.json text:
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
}
"""

# RMSNorm layer
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

# RoPE functions
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

# model functions

# used for GQA
def repeat_kv(x):
    # GQA: expand [batch, num_kv_heads, seq, head_dim] -> [batch, num_kv_heads*N_REP, seq, head_dim]
    if N_REP == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, N_REP, s, d).reshape(b, h * N_REP, s, d)

# singular attention head
class QwenAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=True)
        self.k_proj = torch.nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=True)
        self.v_proj = torch.nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=True)
        self.o_proj = torch.nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # cached kv
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_kv = (k, v)

        # grouped-query attention (GQA): repeat each kv head for its group of query heads
        k = repeat_kv(k)
        v = repeat_kv(v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None))
        out = out.transpose(1, 2).reshape(B, T, NUM_HEADS * HEAD_DIM)
        return self.o_proj(out), present_kv

# MLP
class QwenMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = torch.nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = torch.nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) # uses silu as hidden activation

# attention block
class QwenBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
        self.self_attn = QwenAttention()
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
        self.mlp = QwenMLP()

    def forward(self, x, cos, sin, past_kv = None):
        attention_output, present_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + attention_output
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_kv

# full Qwen model
class Qwen2ForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList(QwenBlock() for _ in range(NUM_LAYERS))
        self.norm = RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)

        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope_cos_sin(self, seq_len, past_len, device, dtype):
        t = torch.arange(past_len, past_len + seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)

    def forward(self, input_ids, full_past_kv):
        x = self.embed_tokens(input_ids)
        past_len = full_past_kv[0][0].shape[2] if full_past_kv is not None else 0
        cos, sin = self.rope_cos_sin(input_ids.shape[1], past_len, x.device, x.dtype)

        if full_past_kv is None:
            full_past_kv = [None] * NUM_LAYERS

        full_present_kv = []
        for layer, past_kv in zip(self.layers, full_past_kv):
            x, present_kv = layer(x, cos, sin, past_kv)
            full_present_kv.append(present_kv)

        x = self.norm(x)
        logits = x @ self.embed_tokens.weight.T
        return logits, full_present_kv