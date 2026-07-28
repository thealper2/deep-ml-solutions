def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    Sd = alpha[:, None] * S
    readout = k @ Sd
    erase = beta * np.outer(k, readout)
    write = beta * np.outer(k, v)
    return Sd - erase + write
