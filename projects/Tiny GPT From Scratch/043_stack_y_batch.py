import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    batch = []
    for offset in offsets:
        batch.append(slice_y_at_offset(data, offset, block_size))

    return np.array(batch)
