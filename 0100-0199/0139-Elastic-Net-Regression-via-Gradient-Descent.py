import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(max_iter):
        y_pred = X @ w + b
        error = y_pred - y

        grad_w  = (1 / n_samples) * (X.T @ error)
        grad_b = np.mean(error)

        grad_w += alpha1 * np.sign(w)
        grad_w += 2 * alpha2 * w

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if np.linalg.norm(grad_w, 1) < tol:
            break

    return np.round(w, 2), round(b, 2)