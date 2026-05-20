def kv_cache_memory(batch_size: int, seq_len: int, embed_dim: int, n_heads: int, n_layers: int, bytes_per_elem: int) -> list:
    """
    Return [mha_bytes, linear_bytes, mixed_bytes] for the three architectures.
    Mixed uses a 3:1 ratio of linear-attention layers to MHA layers.
    """
    d_head = embed_dim // n_heads
    mha_per_layer = 2 * batch_size * seq_len * n_heads * d_head * bytes_per_elem
    mha_bytes = mha_per_layer * n_layers
    linear_per_layer = batch_size * n_heads * d_head * d_head * bytes_per_elem
    linear_bytes = linear_per_layer * n_layers
    layers_per_block = 4
    n_blocks = n_layers // layers_per_block
    mha_layers_per_block = 1
    linear_layers_per_block = 3
    mha_layers_total = n_blocks * mha_layers_per_block
    linear_layers_total = n_blocks * linear_layers_per_block
    mixed_bytes = (mha_per_layer * mha_layers_total) + (linear_per_layer * linear_layers_total)
    return [mha_bytes, linear_bytes, mixed_bytes]