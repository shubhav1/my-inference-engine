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

    def forward(self, x, cos, sin, mask, is_causal, cache, layer_idx, slot_mapping, gather_idx):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # scatter this step's new k/v into their blocks in the paged cache, then gather every
        # position so far back into a dense [B, kv_heads, S, head_dim] tensor for SDPA
        cache.write(layer_idx, slot_mapping, k, v)
        k, v = cache.read(layer_idx, gather_idx)

        # grouped-query attention (GQA): repeat each kv head for its group of query heads
        k = repeat_kv(k)
        v = repeat_kv(v)

        # mask is None whenever every row is a plain (unpadded) causal sequence -- let SDPA
        # use its fused kernel (is_causal) instead of materializing an explicit attn_mask
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, NUM_HEADS * HEAD_DIM)
        return self.o_proj(out)

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

    def forward(self, x, cos, sin, mask, is_causal, cache, layer_idx, slot_mapping, gather_idx):
        attention_output = self.self_attn(
            self.input_layernorm(x), cos, sin, mask, is_causal,
            cache, layer_idx, slot_mapping, gather_idx,
        )
        x = x + attention_output
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# full Qwen model
class Qwen2ForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList(QwenBlock() for _ in range(NUM_LAYERS))
        self.norm = RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)

        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope_cos_sin(self, position_ids, dtype):
        # position_ids: [B, T], one RoPE position per row (rows can differ, e.g. batched decode)
        freqs = position_ids.to(torch.float32)[..., None] * self.inv_freq[None, None, :]  # [B, T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, head_dim]
        return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)  # [B, 1, T, head_dim]

    def forward(self, input_ids, cache, seqs):
        """input_ids: [B, T]. cache: a KVCache. seqs: that cache's sequence id for each batch
        item (list[int]), every row's prior length is tracked by the cache itself.
        """
        _, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        # each row's valid length before this step (plain ints, no device sync), plus the
        # physical block slots this step writes to and reads from, built once for all layers
        starts_list, slot_mapping, gather_idx = cache.prepare(seqs, T)
        starts = torch.tensor(starts_list, device=x.device)

        # RoPE position per row, offset by how much of its cache is already filled
        position_ids = starts[:, None] + torch.arange(T, device=x.device)[None, :]
        cos, sin = self.rope_cos_sin(position_ids, x.dtype)

        # skip building an explicit mask when every row shares the same start, SDPA's
        # fused is_causal kernel is exact then (True only for a fresh prefill; PyTorch's
        # is_causal is top-left aligned, so a T==1 decode step needs mask=None instead)
        same_start = all(s == starts_list[0] for s in starts_list)
        c0 = starts_list[0]
        if same_start and (T == 1 or c0 == 0):
            mask = None
            is_causal = (c0 == 0)
        else:
            # ragged batch: build the real bottom-right causal mask from each row's own start
            S = max(starts_list) + T
            key_pos = torch.arange(S, device=x.device)
            query_pos = starts[:, None] + torch.arange(T, device=x.device)[None, :]  # [B, T]
            mask = (key_pos[None, None, :] <= query_pos[:, :, None]).unsqueeze(1)  # [B, 1, T, S]
            is_causal = False

        # run every layer, each writing this step's k/v into the cache as it goes
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, is_causal, cache, layer_idx, slot_mapping, gather_idx)

        # advance every row's length once, now that all layers have written this step
        cache.advance(seqs, T)

        x = self.norm(x)
        logits = x @ self.embed_tokens.weight.T
        return logits