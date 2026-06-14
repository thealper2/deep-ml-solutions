def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    x = cache['x']
    w1 = cache['w1']
    h1 = cache['h1']
    a1 = cache['a1']
    w2 = cache['w2']
    
    B, T, d_ff = a1.shape
    _, _, d_model = x.shape
    
    d_a1 = d_out @ w2.T
    dw2 = a1.reshape(-1, d_ff).T @ d_out.reshape(-1, d_model)
    db2 = np.sum(d_out, axis=(0, 1))
    d_h1 = d_a1 * (h1 > 0)
    dx = d_h1 @ w1.T
    dw1 = x.reshape(-1, d_model).T @ d_h1.reshape(-1, d_ff)
    db1 = np.sum(d_h1, axis=(0, 1))

    return {'dx': dx, 'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
