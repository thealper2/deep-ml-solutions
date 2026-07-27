def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    out = sigmoid(z)
    return out
