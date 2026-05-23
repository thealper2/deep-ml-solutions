import numpy as np

def smooth_labels(y_true, num_classes, epsilon):
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: Iterable[int] of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        np.ndarray of shape (N, K) with smoothed probabilities.
    """
    N = len(y_true)
    y_true = np.array(y_true)
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), y_true] = 1.0
    smoothed = one_hot * (1 - epsilon) + epsilon / num_classes
    return smoothed


def label_smoothing_cross_entropy(logits, y_true, num_classes, epsilon=0.1, round_decimals=None):
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: Array-like of shape (N, K), model output scores.
        y_true: Array-like of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """
    logits = np.array(logits)
    y_true = np.array(y_true)
    N, K = logits.shape

    max_logits = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
    smoothed_targets = smooth_labels(y_true, num_classes, epsilon)
    loss = -np.sum(smoothed_targets * log_probs, axis=1)
    mean_loss = np.mean(loss)
    if round_decimals is not None:
        mean_loss = round(mean_loss, round_decimals)

    return mean_loss