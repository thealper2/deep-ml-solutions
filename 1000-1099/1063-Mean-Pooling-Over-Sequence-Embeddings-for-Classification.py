import numpy as np

def mean_pool(logits: np.ndarray) -> list:
    """
    Mean-pool per-token logits over the sequence dimension.

    Args:
        logits: array of shape (batch_size, seq_len, num_classes)

    Returns:
        Nested list of shape (batch_size, num_classes), rounded to 4 decimals.
    """
    mean = np.mean(logits, axis=1)
    return mean.tolist()