import numpy as np

def diff(a: np.ndarray) -> np.ndarray:
    """out[0]=a[0]; out[i]=a[i]-a[i-1] for i>0."""
    out = np.zeros_like(a)
    out[0] = a[0]
    out[1:] = a[1:] - a[:-1]
    return out
