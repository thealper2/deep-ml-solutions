import numpy as np

def tanh_backward(h: np.ndarray, dh: np.ndarray) -> np.ndarray:
    """
    Backward pass for tanh using the cached output h = tanh(z).

    Args:
        h:  numpy array, cached output of tanh in the forward pass
        dh: numpy array, upstream gradient dL/dh (same shape as h)

    Returns:
        dz: numpy array, gradient dL/dz (same shape as h)
    """
    z = 1 - h**2
    m = np.multiply(z, dh)
    return m