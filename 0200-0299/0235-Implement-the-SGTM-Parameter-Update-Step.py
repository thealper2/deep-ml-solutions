import numpy as np


def sgtm_step(
    params: np.ndarray,
    grad: np.ndarray,
    forget_mask: np.ndarray,
    lr: float = 0.1,
    batch_type: str = "forget",
) -> np.ndarray:
    """Perform one SGTM update step on a 1D parameter vector.

    SGTM splits parameters into 'forget' and 'retain' groups using a binary mask.
    Depending on the batch type, only some parameters are allowed to update.

    Rules:
    - 'forget' batch: only forget parameters update (retain params get zeroed gradients).
    - 'retain' batch: only retain parameters update (forget params get zeroed gradients).
    - 'unlabeled' batch: all parameters update normally.

    Any nonzero entry in forget_mask should be treated as 1 (forget), and zero
    means retain.

    Args:
        params: Current parameters, shape (d,)
        grad: Gradient for this batch, shape (d,)
        forget_mask: Mask for forget parameters, shape (d,)
        lr: Learning rate
        batch_type: One of {'forget', 'retain', 'unlabeled'}

    Returns:
        new_params: Updated parameters
    """

    masked_grad = grad.copy()

    if batch_type == 'forget':
        forget_binary = (forget_mask != 0).astype(float)
        masked_grad = grad * forget_binary

    elif batch_type == 'retain':
        forget_binary = (forget_mask != 0).astype(float)
        retain_binary = 1 - forget_binary
        masked_grad = grad * retain_binary

    elif batch_type == 'unlabeled':
        pass
    
    else:
        raise ValueError('Invalid batch_type')

    new_params = params - lr * masked_grad
    return new_params.tolist()