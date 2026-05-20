import numpy as np

def reduce_broadcast_grad(upstream_grad: np.ndarray, operand_shape: tuple) -> np.ndarray:
    """
    Reduce an upstream gradient back to the shape of a broadcasted operand.

    Args:
        upstream_grad: array of shape (N, D)
        operand_shape: either (1, D) or (N, 1)

    Returns:
        numpy array with shape exactly equal to operand_shape
    """
    if operand_shape == (1, upstream_grad.shape[1]):
        return np.sum(upstream_grad, axis=0, keepdims=True)
    elif operand_shape == (upstream_grad.shape[0], 1):
        return np.sum(upstream_grad, axis=1, keepdims=True)
    else:
        raise ValueError("Invalid operand_shape")
