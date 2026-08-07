import numpy as np

def recursive_belief(e, B0, gamma: float = 0.95, eps: float = 1e-4) -> dict:
    """
    Returns dict with keys 'c', 'ell', 'B' — each a float array of shape (K,).
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
