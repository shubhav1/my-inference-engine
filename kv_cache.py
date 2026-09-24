import math

import torch

from model import HEAD_DIM, NUM_KV_HEADS, NUM_LAYERS


class KVCache:
    """Paged (blocked) KV cache shared by every concurrently active sequence.

    Buffer shape: [NUM_LAYERS, 2, num_blocks * block_size, NUM_KV_HEADS, HEAD_DIM] (dim 1: k=0,
    v=1). The token dimension is a flat pool carved into fixed-size blocks, never resized after
    construction. A sequence owns a list of blocks (its block table), grown one block at a time
    as it gets longer, so it only ever holds memory for tokens it actually has -- no per-sequence
    worst-case reservation. Logical position p of a sequence lives at physical token slot
    block_table[p // block_size] * block_size + p % block_size.
    """

    def __init__(self, num_blocks, block_size, device, dtype):
        self.buf = torch.zeros(
            NUM_LAYERS, 2, num_blocks * block_size, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype
        )
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self._free_blocks = list(range(num_blocks))
        self._block_tables = {}  # seq id -> list of block ids, in logical order
        self._lengths = {}  # seq id -> tokens written so far (plain ints: reading them never syncs the device)
        self._next_seq = 0

    def free_block_count(self):
        return len(self._free_blocks)

    def blocks_needed(self, seq, total_len):
        """Extra blocks `seq` (None for a brand-new sequence) needs to hold total_len tokens."""
        held = len(self._block_tables[seq]) if seq is not None else 0
        return max(0, math.ceil(total_len / self.block_size) - held)

    def alloc(self):
        """Start a new, empty sequence. Blocks are only taken as it gets written to."""
        seq = self._next_seq
        self._next_seq += 1
        self._block_tables[seq] = []
        self._lengths[seq] = 0
        return seq

    def free(self, seq):
        """Return every block this sequence holds to the pool."""
        self._free_blocks.extend(self._block_tables.pop(seq))
        del self._lengths[seq]

    def length(self, seq):
        return self._lengths[seq]

    def ensure_capacity(self, seq, total_len):
        """Grow seq's block table until it covers total_len tokens. Raises if the pool runs dry."""
        need = self.blocks_needed(seq, total_len)
        if need > len(self._free_blocks):
            raise RuntimeError(
                f"KV cache exhausted: seq {seq} needs {need} more blocks, only {len(self._free_blocks)} free"
            )
        table = self._block_tables[seq]
        for _ in range(need):
            table.append(self._free_blocks.pop())

    def trim(self, seq):
        """Give back any blocks past the one holding seq's last written token (e.g. what was
        reserved up front for a reply that ended early)."""
        table = self._block_tables[seq]
        keep = math.ceil(self._lengths[seq] / self.block_size)
        self._free_blocks.extend(table[keep:])
        del table[keep:]

    def prepare(self, seqs, T):
        """Set up one forward step that appends T tokens to every seq in `seqs`.

        Returns (starts, slot_mapping, gather_idx): each row's length before this step as plain
        ints, the physical slot of every new token ([B*T], row-major, for write()), and the
        physical slot of every logical position [0, max(starts) + T) of every row ([B, S], for
        read()). Positions past a row's own length point at slot 0, whatever is there is garbage
        the attention mask already hides.
        """
        bs = self.block_size
        starts = [self._lengths[s] for s in seqs]
        for s, start in zip(seqs, starts):
            self.ensure_capacity(s, start + T)

        slot_mapping = []
        for s, start in zip(seqs, starts):
            table = self._block_tables[s]
            slot_mapping.extend(table[p // bs] * bs + p % bs for p in range(start, start + T))

        S = max(starts) + T
        n_blocks = math.ceil(S / bs)
        tables = [(self._block_tables[s] + [0] * n_blocks)[:n_blocks] for s in seqs]
        tables = torch.tensor(tables, dtype=torch.long).to(self.device, non_blocking=True)
        pos = torch.arange(S, device=self.device)
        gather_idx = tables[:, pos // bs] * bs + pos % bs  # [B, S]

        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long).to(self.device, non_blocking=True)
        return starts, slot_mapping, gather_idx

    def advance(self, seqs, T):
        """Record that every seq in `seqs` now holds T more tokens (after all layers wrote)."""
        for s in seqs:
            self._lengths[s] += T

    def write(self, layer_idx, slot_mapping, k, v):
        """Scatter this step's new k/v for one layer into their physical slots, in one op.

        k, v: [B, NUM_KV_HEADS, T, HEAD_DIM], flattened row-major to match slot_mapping.
        """
        B, H, T, D = k.shape
        self.buf[layer_idx, 0].index_copy_(0, slot_mapping, k.transpose(1, 2).reshape(B * T, H, D))
        self.buf[layer_idx, 1].index_copy_(0, slot_mapping, v.transpose(1, 2).reshape(B * T, H, D))

    def read(self, layer_idx, gather_idx):
        """Gather this layer's k/v into dense [B, NUM_KV_HEADS, S, HEAD_DIM] tensors (a copy)."""
        k = self.buf[layer_idx, 0][gather_idx].transpose(1, 2)
        v = self.buf[layer_idx, 1][gather_idx].transpose(1, 2)
        return k, v
