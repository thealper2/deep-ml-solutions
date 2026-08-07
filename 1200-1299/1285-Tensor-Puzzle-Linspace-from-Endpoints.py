import numpy as np

def linspace(start, stop, n: int) -> np.ndarray:
    """n evenly spaced values from start to stop inclusive."""
    if n == 1:
        return np.array([start])

    step = (stop - start) / (n - 1)
    out = start + np.arange(n) * step
    return out
