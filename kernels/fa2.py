import triton
import triton.language as tl
import torch
import math

def get_configs():
    # winners of a full 144-config search (Br, Bc in 16..128, 2-8 warps, 2-4 stages) on an A10G,
    # bf16, head_dim 128. autotune just picks between them per key, re-run the full search on a new GPU
    return [
        triton.Config({"Br": 32, "Bc": 64}, num_warps=4, num_stages=2),  # prefill (causal)
        triton.Config({"Br": 16, "Bc": 64}, num_warps=4, num_stages=3),  # decode (non-causal)
    ]

@triton.autotune(
    configs=get_configs(),
    key=["d", "d_v", "is_causal"],  # not seq lens: they change every decode step
    cache_results=True,  # save the winning config to TRITON_CACHE_DIR so later runs skip tuning
)

# args that change with sequence length: without this, triton compiles a new variant whenever one
# crosses a multiple of 16 (e.g. mid-decode), stalling that token for the length of a compile
@triton.jit(do_not_specialize=["SEQLEN_Q", "SEQLEN_K", "Lb_stride", "Lh_stride"])
def fa2_forward(
    # inputs
    Q_ptr, K_ptr, V_ptr,
    Qb_stride, Qh_stride, Qs_stride, Qd_stride,
    Kb_stride, Kh_stride, Ks_stride, Kd_stride,
    Vb_stride, Vh_stride, Vs_stride, Vd_stride,
    sm_scale,
    KV_lens_ptr,  # [batch] int32: valid keys per row, keys past it are padding (ragged batches)
    # ouputs
    O_ptr, L_ptr,
    Ob_stride, Oh_stride, Os_stride, Od_stride,
    Lb_stride, Lh_stride, Ls_stride,
    # constexprs
    Br: tl.constexpr, Bc: tl.constexpr, 
    d: tl.constexpr, d_v: tl.constexpr,
    is_causal: tl.constexpr, SEQLEN_Q, SEQLEN_K,
    GROUP: tl.constexpr,  # query heads per kv head (GQA), 1 for plain MHA
):
    # program ids
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    # tl.device_print("tile_id", tile_id)
    # tl.device_print("batch_id", batch_id)
    # tl.device_print("head_id", head_id)

    q_offset = batch_id.to(tl.int64) * Qb_stride + head_id.to(tl.int64) * Qh_stride
    
    # GQA: each group of GROUP consecutive query heads reads the same kv head
    kv_head_id = head_id // GROUP
    k_offset = batch_id.to(tl.int64) * Kb_stride + kv_head_id.to(tl.int64) * Kh_stride
    v_offset = batch_id.to(tl.int64) * Vb_stride + kv_head_id.to(tl.int64) * Vh_stride

    o_offset = batch_id.to(tl.int64) * Ob_stride + head_id.to(tl.int64) * Oh_stride 
    l_offset = batch_id.to(tl.int64) * Lb_stride + head_id.to(tl.int64) * Lh_stride 

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + q_offset,
        shape = (SEQLEN_Q, d),
        strides = (Qs_stride, Qd_stride),
        offsets = (Br * tile_id, 0),
        block_shape = (Br, d),
        order = (1,0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + k_offset,
        shape = (SEQLEN_K, d),
        strides = (Ks_stride, Kd_stride),
        offsets = (0, 0),
        block_shape = (Bc, d),
        order = (1,0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + v_offset,
        shape = (SEQLEN_K, d_v),
        strides = (Vs_stride, Vd_stride),
        offsets = (0, 0),
        block_shape = (Bc, d_v),
        order = (1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + o_offset,
        shape = (SEQLEN_Q, d_v),
        strides = (Os_stride, Od_stride),
        offsets = (Br * tile_id, 0),
        block_shape = (Br, d_v),
        order = (1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + l_offset,
        shape = (SEQLEN_Q,),
        strides = (Ls_stride,),
        offsets = (Br * tile_id,),
        block_shape = (Br,),
        order = (0,)
    )

    i_start = tile_id * Br
    # this row's own key count: K/V are padded to SEQLEN_K across the batch, only kv_len are real
    kv_len = tl.load(KV_lens_ptr + batch_id)

    # causal mask is bottom-right aligned per row: query i sits at key position i + causal_offset
    causal_offset = kv_len - SEQLEN_Q
    i_end = tl.minimum(i_start + Br, SEQLEN_Q) + causal_offset

    # load Qi from HBM to SRAM
    Qi = tl.load(Q_block_ptr, boundary_check = (0, 1))

    # initialize Oi, li, mi
    Oi = tl.zeros((Br, d_v), dtype=tl.float32)
    li = tl.zeros((Br,), dtype=tl.float32)
    mi = tl.full((Br,), float('-inf'), dtype=tl.float32)

    #j_max = SEQLEN if not is_causal else i_end
    for j_start in range(0, kv_len, Bc):
        if not is_causal or (i_end > j_start):
            # load Kj, Vj from HBM to SRAM
            Kj = tl.load(K_block_ptr, boundary_check = (0,1))
            Vj = tl.load(V_block_ptr, boundary_check = (0,1))

            # compute Sij
            Sij = tl.dot(Qi, tl.trans(Kj), input_precision="ieee") * sm_scale # preventing tf32 to test accuracy w fp32 cpu execution

            # masking causal/out of bound indices
            j_idx = j_start + tl.arange(0, Bc)
            valid = j_idx[None, :] < kv_len
            if is_causal:
                i_idx = i_start + causal_offset + tl.arange(0, Br)
                valid = valid & (i_idx[:, None] >= j_idx[None, :])

            Sij = tl.where(valid, Sij, float("-inf"))

            # compute mij, Pij, lij
            mij = tl.maximum(mi, tl.max(Sij, axis=1)) # tl.maximum takes elementwise max, tl.max reduces tensor along axis
            Pij = tl.exp(Sij - mij[:, None])
            rescale = tl.exp(mi - mij)
            li = rescale * li + tl.sum(Pij, axis=1)
            Oi = rescale[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj, input_precision="ieee") # preventing tf32 to test accuracy w fp32 cpu execution

            # update mi
            mi = mij

            # advance pointers (keeping in loop bc further advances would also be invalid under causal masking)
            K_block_ptr = tl.advance(K_block_ptr, (Bc, 0))
            V_block_ptr = tl.advance(V_block_ptr, (Bc, 0))

    # compute scaled Oi
    Oi = Oi / li[:, None]

    # compute Li (logsumexp)
    Li = mi + tl.log(li)

    # write Oi and Li to HBM
    tl.store(O_block_ptr, Oi, boundary_check = (0,1))
    tl.store(L_block_ptr, Li, boundary_check = (0,))

class FlashAttention2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False, kv_lens = None):
        # kv_lens: optional [batch] valid key count per row, for batches whose K/V are padded to
        # different lengths. Defaults to every row using all of K/V
        # calls fa2 forward triton function

        # Bc = Br = 16
        batch_size, num_heads, seq_len, d = Q.shape
        seq_len_k = K.shape[2]
        d_v = V.shape[-1]
        sm_scale = 1.0 / math.sqrt(d)

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        num_kv_heads = K.shape[1]
        assert Q.shape[0] == K.shape[0] and Q.shape[-1] == K.shape[-1], "Q and K match in batch and head dim"
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
        assert K.shape[:-1] == V.shape[:-1], "K and V are the same shape for all dimensions but the last"
        if kv_lens is None:
            kv_lens = torch.full((batch_size,), seq_len_k, device=Q.device, dtype=torch.int32)
        assert kv_lens.shape == (batch_size,) and kv_lens.dtype == torch.int32, "kv_lens is [batch] int32"
        assert (d & (d - 1) == 0) and (d_v & (d_v - 1) == 0), "d and d_v must be powers of 2 (Triton block_shape constraint)"

        # initialize output tensors O & L
        O = torch.empty((batch_size, num_heads, seq_len, d_v), device=Q.device, dtype = Q.dtype)
        L = torch.empty((batch_size, num_heads, seq_len), device=Q.device, dtype = torch.float32)

        # call function
        grid = lambda META: (triton.cdiv(seq_len, META["Br"]), batch_size, num_heads)
        fa2_forward[grid](
            Q, K, V,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            sm_scale,
            kv_lens,
            O, L,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            # Br=Br, Bc=Bc,
            d=d, d_v=d_v, 
            is_causal=is_causal, SEQLEN_Q=seq_len, SEQLEN_K=seq_len_k,
            GROUP=num_heads // num_kv_heads,
        )

        # update context
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = sm_scale

        return O