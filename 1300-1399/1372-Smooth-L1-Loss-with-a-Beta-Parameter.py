import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, get {beta}")

    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    x = pred - target
    abs_x = np.abs(x)

    loss = np.where(
        abs_x < beta,
        0.5 * x * x / beta,
        abs_x - 0.5 * beta
    )

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")