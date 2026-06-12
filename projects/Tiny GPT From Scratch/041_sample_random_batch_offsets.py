def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    max_start = data_len - block_size - 1
    offsets = rng.integers(0, max_start, size=batch_size)
    return offsets
