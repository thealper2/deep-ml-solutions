def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    return default_size if default_size > 1 else 1
