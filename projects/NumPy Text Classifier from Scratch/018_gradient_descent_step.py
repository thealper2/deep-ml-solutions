def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    m = X.shape[0]
    probs = logistic_predict_proba(X, w, b)
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1 - eps)
    loss = binary_cross_entropy(y, probs_clipped, w, l2_lambda)
    dw = (1 / m) * X.T @ (probs - y) + l2_lambda * w
    db = (1 / m) * np.sum(probs - y)
    w_new = w - lr * dw
    b_new = b - lr * db
    return w_new, b_new, loss
