def compute_d_head(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError('d_model % n_heads != 0')
    return d_model // n_heads
