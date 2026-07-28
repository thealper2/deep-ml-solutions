def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    q_proj = x @ params['Wq']
    q_conv = short_conv(q_proj, params['cq'])
    q_swish = q_conv * (1 / (1 + np.exp(-q_conv)))
    q = q_swish / np.sqrt(np.sum(q_swish ** 2, axis=1, keepdims=True) + 1e-6)

    k_proj = x @ params['Wk']
    k_conv = short_conv(k_proj, params['ck'])
    k_swish = k_conv * (1 / (1 + np.exp(-k_conv)))
    k = k_swish / np.sqrt(np.sum(k_swish ** 2, axis=1, keepdims=True) + 1e-6)

    v_proj = x @ params['Wv']
    v_conv = short_conv(v_proj, params['cv'])
    v = v_conv * (1 / (1 + np.exp(-v_conv)))
    return q, k, v
