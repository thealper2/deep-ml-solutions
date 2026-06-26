import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    N, C = preds.shape

    eps = 1e-12
    preds_clipped = np.clip(preds, eps, 1.0 - eps)

    one_hot = np.zeros_like(preds_clipped)
    one_hot[np.arange(N), target] = 1.0

    per_sample_loss = -np.sum(one_hot * np.log(preds_clipped), axis=1)

    grad = (preds_clipped - one_hot) / N if reduction == 'mean' else (preds_clipped - one_hot)

    if reduction == 'mean':
        loss = np.mean(per_sample_loss)
    elif reduction == 'sum':
        loss = np.sum(per_sample_loss)
    else:
        loss = per_sample_loss
        grad = preds_clipped - one_hot

    return loss, grad
