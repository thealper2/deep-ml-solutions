import torch

def sgd_step(w, grad, lr):
    """Perform one SGD update step.

    Args:
        w: Current parameter tensor.
        grad: Gradient tensor (same shape as w).
        lr: Learning rate (float).

    Returns:
        Updated parameter tensor w - lr * grad.
    """
    return w - lr * grad
