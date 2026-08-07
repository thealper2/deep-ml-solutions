import numpy as np

def recursive_belief(e, B0, gamma=0.95, eps=1e-4):
    """
    Compute belief trajectory from turn-level evidence.
    
    Args:
        e: Turn-level evidence of shape (K,)
        B0: Prior belief in (0, 1)
        gamma: Decay factor for evidence accumulation
        eps: Numerical safety for logit clipping
    
    Returns:
        dict with keys 'c', 'ell', 'B' each as NumPy arrays of shape (K,)
    """
    B0_clipped = np.clip(B0, eps, 1.0 - eps)
    ell0 = np.log(B0_clipped / (1.0 - B0_clipped))
    
    K = len(e)
    c = np.zeros(K)
    ell = np.zeros(K)
    B = np.zeros(K)
    
    c_prev = 0.0
    for k in range(K):
        c_k = gamma * c_prev + e[k]
        ell_k = ell0 + c_k
        B_k = 1.0 / (1.0 + np.exp(-ell_k))
        
        c[k] = c_k
        ell[k] = ell_k
        B[k] = B_k
        
        c_prev = c_k
    
    return {'c': c, 'ell': ell, 'B': B}


def bounded_reshape_weights(q, b=0.2, eps=1e-4):
    """
    Compute bounded per-turn multipliers from outcome-aligned credits.
    
    Args:
        q: Outcome-aligned credits of shape (K,)
        b: Bound parameter (default 0.2)
        eps: Numerical safety for variance
    
    Returns:
        w: Bounded reshape weights of shape (K,)
    """
    q = np.array(q)
    K = len(q)
    
    if K <= 1:
        return np.ones(K)
    
    mu_q = np.mean(q)
    sigma_q = np.std(q, ddof=0)
    
    if sigma_q == 0:
        return np.ones(K)
    
    z = (q - mu_q) / (sigma_q + eps)
    w = np.clip(1 + b * z, 1 - b, 1 + b)
    
    return w

def agentopsd_turn_advantages(
    turn_evidence,
    A_seq,
    B0,
    gamma: float = 0.95,
    b: float = 0.2,
    lam: float = 0.5,
    eps: float = 1e-4,
):
    """Full AgentOPSD credit assignment → (K,) reshaped advantages."""
    K = len(turn_evidence)
    result = recursive_belief(turn_evidence, B0, gamma=gamma, eps=eps)
    B = result['B']
    B0_clipped = np.clip(B0, eps, 1.0 - eps)
    B_extended = np.concatenate([[B0_clipped], B])
    delta_B = np.diff(B_extended)
    q = np.sign(A_seq) * delta_B
    w = bounded_reshape_weights(q, b=b, eps=eps)
    A_tilde = A_seq * ((1 - lam) + lam * w)
    return A_tilde
