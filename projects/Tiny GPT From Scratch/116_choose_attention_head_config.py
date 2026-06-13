def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    if d_model % n_heads != 0:
        raise ValueError('d_model % n_heads != 0')
    d_head = d_model // n_heads
    return {
        'n_heads': n_heads,
        'd_head': d_head,
        'd_model': d_model,
    }
