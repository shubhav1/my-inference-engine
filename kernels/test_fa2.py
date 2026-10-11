# Checks the FA2 kernel against SDPA on Qwen's attention shapes (12 q heads, 2 kv heads, head_dim 128).
# Needs CUDA: uv run modal run modal_run.py::test_fa2

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # kernels/ and model.py live one level up

from kernels.fa2 import FlashAttention2Function
from model import HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, repeat_kv

# (name, T queries, each row's key count, is_causal), mirroring how model.py calls attention.
# K/V are padded to the longest row, as cache.read() does for a ragged batch
CASES = [
    ("prefill", 300, [300, 300], True),
    ("decode", 1, [1000, 1000], False),
    ("chunked", 64, [500, 500], True),
    ("ragged decode", 1, [1000, 37, 512, 1], False),  # concurrent chats at different lengths
    ("ragged prefill", 16, [16, 300, 100], True),  # new turns appended to chats of different lengths
]
PAD_VALUE = 100.0  # garbage in the padding, so any key that leaks through the mask shows up as a big error
TOLERANCE = {torch.bfloat16: 2e-2}


def reference(q, k, v, is_causal):
    # fp32 SDPA with an explicit bottom-right causal mask (query i sits at key position i + S - T)
    q, k, v = q.float(), repeat_kv(k.float()), repeat_kv(v.float())
    T, S = q.shape[2], k.shape[2]
    mask = None
    if is_causal:
        mask = torch.arange(S, device=q.device)[None, :] <= torch.arange(T, device=q.device)[:, None] + (S - T)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def main():
    torch.manual_seed(0)
    failed = 0
    for dtype in TOLERANCE:  # bf16 only
        for name, T, lens, is_causal in CASES:
            B, S = len(lens), max(lens)
            q = torch.randn(B, NUM_HEADS, T, HEAD_DIM, device="cuda", dtype=dtype)
            k = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=dtype)
            v = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=dtype)
            for b, n in enumerate(lens):
                k[b, :, n:] = PAD_VALUE
                v[b, :, n:] = PAD_VALUE
            kv_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)

            out = FlashAttention2Function.apply(q, k, v, is_causal, kv_lens)
            # reference: each row on its own, cut to its real keys
            err = max(
                (out[b:b + 1].float() - reference(q[b:b + 1], k[b:b + 1, :, :n], v[b:b + 1, :, :n], is_causal)).abs().max().item()
                for b, n in enumerate(lens)
            )
            ok = err < TOLERANCE[dtype]
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {str(dtype):15} {name:15} T={T:<4} kv_lens={lens!s:22} causal={is_causal!s:5}  max err {err:.2e}")
    print("all passed" if not failed else f"{failed} case(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
