import numpy as np

def epsilon_insensitive_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> float:
    """
    Compute the average epsilon-insensitive loss for SVM regression.

    Args:
        y_true: Array of true target values
        y_pred: Array of predicted target values
        epsilon: Non-negative insensitivity tolerance

    Returns:
        Average epsilon-insensitive loss rounded to 4 decimal places
    """
    per_sample_losses = np.maximum(0, np.abs(y_true - y_pred) - epsilon)
    loss = np.mean(per_sample_losses)
    return round(float(loss), 4)
