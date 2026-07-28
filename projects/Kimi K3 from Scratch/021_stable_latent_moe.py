def stable_latent_moe(x, params):
    """y = sum_shared SiTU(x) + RMSNorm(routed_aggregate) @ Wup (Eq. 11).

    Route on full-width x; compute in latent width; RMSNorm(u) before Wup.
    """
    d = x.shape[1]
    l = params['Wdown'].shape[1]
    z = x @ params['Wdown']
    s, idx, p = route_topk(x, params['Wr'], params['bias'], params['k'])
    u = routed_experts(z, idx, p, params['experts'])
    rms = np.sqrt(np.mean(u**2, axis=1, keepdims=True) + 1e-6)
    u_norm = u / rms
    shared_sum = np.zeros((x.shape[0], d))
    for Wg, Wu in params['shared']:
        shared_sum += situ_glu(x, Wg, Wu)
    
    y = shared_sum + u_norm @ params['Wup']
    return y
