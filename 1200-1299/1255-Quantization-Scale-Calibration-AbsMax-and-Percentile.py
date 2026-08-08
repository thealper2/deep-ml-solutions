import numpy as np

def absmax_scale(x: np.ndarray, bits: int = 8) -> float:
    """Symmetric absmax scale: max|x| / qmax, qmax=2^(bits-1)-1."""
    qmax = 2 ** (bits - 1) - 1
    max_val = np.max(x)
    if max_val == 0:
        return 1.0

    return np.max(np.abs(x)) / qmax

def percentile_scale(x: np.ndarray, bits: int = 8, p: float = 99.9) -> float:
    """Symmetric percentile scale on |x|."""
    qmax = 2 ** (bits - 1) - 1
    if p == 0:
        return 1.0

    return np.percentile(np.abs(x) / qmax, p)
