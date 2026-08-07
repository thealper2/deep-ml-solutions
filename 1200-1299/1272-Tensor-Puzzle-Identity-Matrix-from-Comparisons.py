import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    i = np.arange(n)
    j = np.arange(n)
    return (i[:, None] == j).astype(np.float64)
