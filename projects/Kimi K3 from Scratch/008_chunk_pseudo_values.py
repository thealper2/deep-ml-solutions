def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    C, dk = k.shape
    dv = v.shape[1]

    Gamma = cumulative_decay(alpha)
    Khat = k * Gamma
    Kcheck = k / Gamma

    M = np.tril(Khat @ Kcheck.T, k=-1)

    rhs = beta[:, None] * (v - Khat @ S0)

    L = np.eye(C) + np.diag(beta) @ M
    U = np.linalg.solve(L, rhs)

    return U
