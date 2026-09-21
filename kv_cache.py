import torch

from model import HEAD_DIM, NUM_KV_HEADS, NUM_LAYERS


class KVCache:
    """Preallocated, fixed-shape KV cache shared by every concurrently active sequence.

    Buffer shape: [NUM_LAYERS, 2, max_batch, NUM_KV_HEADS, max_len, HEAD_DIM] (dim 1: k=0, v=1).
    Never resized after construction. Each batch row is a "slot" owned by one sequence;
    a decode step writes that row's new k/v in place at its current length and advances a
    per-slot counter.
    """

    def __init__(self, max_batch, max_len, device, dtype):
        self.buf = torch.zeros(
            NUM_LAYERS, 2, max_batch, NUM_KV_HEADS, max_len, HEAD_DIM, device=device, dtype=dtype
        )
        self.lengths = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.max_batch = max_batch
        self.max_len = max_len
        self._free_slots = list(range(max_batch))

    def free_slot_count(self):
        return len(self._free_slots)

    def alloc(self):
        """Reserve a slot for a new sequence. Raises if every slot is already owned."""
        if not self._free_slots:
            raise RuntimeError(f"KV cache exhausted: all {self.max_batch} slots are in use")
        slot = self._free_slots.pop()
        self.lengths[slot] = 0
        return slot

    def free(self, slot):
        """Release a sequence's slot back to the pool once its chat is done with it."""
        self._free_slots.append(slot)

    def write(self, layer_idx, slots, starts, k, v):
        """Write new k/v for one layer, one step, in place.

        k, v: [B, NUM_KV_HEADS, T, HEAD_DIM]. Row i is written at
        buf[layer_idx, :, slots[i], :, starts[i]:starts[i]+T, :]. Start offsets can differ
        per row (ragged conversation lengths), so this is a small per-row loop, B is the
        number of concurrently active sequences, not the sequence length.
        """
        T = k.shape[2]
        for i, slot in enumerate(slots):
            start = int(starts[i])
            end = start + T
            if end > self.max_len:
                raise RuntimeError(
                    f"KV cache overflow: slot {slot} would reach length {end} > max_len {self.max_len}"
                )
            self.buf[layer_idx, 0, slot, :, start:end, :] = k[i]
            self.buf[layer_idx, 1, slot, :, start:end, :] = v[i]

    def read(self, layer_idx, slots, upto):
        """View (no copy) of this layer's k/v for the given slots, positions [0, upto)."""
        k = self.buf[layer_idx, 0, slots, :, :upto, :]
        v = self.buf[layer_idx, 1, slots, :, :upto, :]
        return k, v
