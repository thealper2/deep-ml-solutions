def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    c = x @ Wc
    K = (c @ Wk_up).reshape(x.shape[0], n_heads, -1)
    V = (c @ Wv_up).reshape(x.shape[0], n_heads, -1)
    return c, K, V
