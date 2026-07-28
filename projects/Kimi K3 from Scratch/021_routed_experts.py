def routed_experts(z, idx, p, experts):
    """u[t] = sum_j p[t,j] * situ_glu(z[t], *experts[idx[t,j]]).

    z: (T, l).  experts: list of (Wg, Wu) latent-width SiTU-GLUs. Returns (T, l).
    """
    T, l = z.shape
    k = idx.shape[1]
    u = np.zeros((T, l))

    for t in range(T):
        for j in range(k):
            expert_idx = idx[t, j]
            Wg, Wu = experts[expert_idx]
            expert_out = situ_glu(z[t:t+1], Wg, Wu)
            u[t] += p[t, j] * expert_out[0]

    return u
