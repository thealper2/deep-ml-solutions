def qkv_projection_backward(d_q, d_k, d_v, cache):
    x = cache['x']
    w_q = cache['w_q']
    w_k = cache['w_k']
    w_v = cache['w_v']

    B, T, d_in = x.shape
    _, _, d_head = d_q.shape

    x_flat = x.reshape(-1, d_in)
    d_q_flat = d_q.reshape(-1, d_head)
    d_k_flat = d_k.reshape(-1, d_head)
    d_v_flat = d_v.reshape(-1, d_head)

    dw_q = x_flat.T @ d_q_flat
    dw_k = x_flat.T @ d_k_flat
    dw_v = x_flat.T @ d_v_flat

    dx_flat = d_q_flat @ w_q.T + d_k_flat @ w_k.T + d_v_flat @ w_v.T
    dx = dx_flat.reshape(B, T, d_in)
    
    return {
        'dx': dx,
        'dw_q': dw_q,
        'dw_k': dw_k,
        'dw_v': dw_v,
    }
