import numpy as np

def bounded_reshape_weights(q, b: float = 0.2, eps: float = 1e-4):
    """Within-trajectory z-score of q, then w = clip(1 + b*z, 1-b, 1+b)."""
    q = np.array(q)
    K = len(q)

    if K <= 1 or np.std(q, ddof=0) == 0:
        return np.ones(K)

    mu_q = np.mean(q)
    sigma_q = np.std(q, ddof=0)

    z = (q - mu_q) / (sigma_q + eps)
    w = np.clip(1 + b * z, 1 - b, 1 + b)
    return w
