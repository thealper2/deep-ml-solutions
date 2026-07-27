def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    return np.where(proba >= threshold, 1, 0)
