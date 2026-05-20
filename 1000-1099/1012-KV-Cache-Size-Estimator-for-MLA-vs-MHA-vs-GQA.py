def kv_cache_bytes(batch: int, seqlen: int, n_layers: int, n_heads: int, head_dim: int, n_kv_groups: int, latent_dim: int, attn_type: str, bytes_per_elem: int = 2) -> int:
    """Return KV cache memory in bytes for the given attention type."""
    if attn_type == 'mha':
        b = 2 * batch * seqlen * n_layers * n_heads * head_dim * bytes_per_elem
    elif attn_type == 'gqa':
        b = 2 * batch * seqlen * n_layers * n_kv_groups * head_dim * bytes_per_elem
    elif attn_type == 'mla':
        b = batch * seqlen * n_layers * latent_dim * bytes_per_elem
    else:
        raise ValueError('Unknown attn_type')

    return b