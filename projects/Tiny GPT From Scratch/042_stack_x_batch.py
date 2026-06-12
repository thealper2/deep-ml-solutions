import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    batch = []
    for offset in offsets:
        batch.append(slice_x_at_offset(data, offset, block_size))

    return np.array(batch)
