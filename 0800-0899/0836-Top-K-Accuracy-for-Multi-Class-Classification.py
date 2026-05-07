import numpy as np

def top_k_accuracy(y_scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    Compute Top-K accuracy for multi-class classification.

    Args:
        y_scores: Predicted scores of shape (n_samples, n_classes)
        y_true: True class labels of shape (n_samples,)
        k: Number of top predictions to consider

    Returns:
        Top-K accuracy as a float rounded to 4 decimal places
    """
    n_samples = y_scores.shape[0]
    correct = 0

    for i in range(n_samples):
        scores = y_scores[i]
        indices = np.argsort(-scores)
        top_k_indices = indices[:k]
        if y_true[i] in top_k_indices:
            correct += 1

    return round(correct / n_samples, 4)