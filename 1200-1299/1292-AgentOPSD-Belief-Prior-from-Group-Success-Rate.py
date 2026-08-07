import numpy as np

def belief_prior(rewards, eps: float = 1e-4) -> float:
    """B0 = clip(mean(rewards), eps, 1-eps)."""
    return np.clip(np.mean(rewards, axis=-1), eps, 1 - eps)
