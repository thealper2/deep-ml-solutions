import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    attn = cache['attn']
    v = cache['v']
    d_attn = d_attn_out @ np.swapaxes(v, -1, -2)
    d_v = np.swapaxes(attn, -1, -2) @ d_attn_out
    return {'d_attn': d_attn, 'd_v': d_v}
