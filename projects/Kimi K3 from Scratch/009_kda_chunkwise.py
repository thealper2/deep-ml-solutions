def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. Returns (O, S_final).
    """
    T, dk = q.shape
    dv = v.shape[1]
    
    if S0 is None:
        S = np.zeros((dk, dv))
    else:
        S = S0.copy()
    
    O = np.zeros((T, dv))
    
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        C = end - start
        
        q_c = q[start:end]
        k_c = k[start:end]
        v_c = v[start:end]
        alpha_c = alpha[start:end]
        beta_c = beta[start:end]
        
        Gamma = cumulative_decay(alpha_c)
        Qhat = q_c * Gamma
        Kcheck = k_c / Gamma
        
        U = chunk_pseudo_values(k_c, v_c, alpha_c, beta_c, S)
        
        inter = Qhat @ S
        
        M = Qhat @ Kcheck.T
        tril_M = np.tril(M)
        intra = tril_M @ U
        
        O[start:end] = inter + intra
        
        S = Gamma[-1][:, None] * (S + Kcheck.T @ U)
    
    return O, S
