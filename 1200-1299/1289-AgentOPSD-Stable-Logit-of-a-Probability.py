import numpy as np

def stable_logit(p, eps: float = 1e-4):
    """Clip p into [eps, 1-eps] then return log(p / (1-p))."""
    p_c = np.clip(p, eps, 1 - eps)
    return np.log(p_c / (1 - p_c))
