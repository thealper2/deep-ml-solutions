import numpy as np

def newton_schulz5(G, steps=5, eps=1e-7):
    M, N = G.shape
    was_transposed = M > N

    if was_transposed:
        X = G.T
    else:
        X = G.copy()

    norm = np.linalg.norm(X, ord='fro')
    if norm > eps:
        X = X / norm
    else:
        X = X * 0

    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        XTX = X.T @ X
        X = a * X + X @ (b * XTX + c * (XTX @ XTX))

    if was_transposed:
        X = X.T

    return X

def muon_step(theta, B, grad, eta, mu, ns_steps=5, eps=1e-7):
    """
    theta: np.ndarray, shape (M, N)
    B: np.ndarray, shape (M, N)
    grad: np.ndarray, shape (M, N)
    eta: float (learning rate)
    mu: float (momentum coefficient)
    ns_steps: int (Newton-Schulz steps)
    eps: float (numerical stability)
    Returns: updated theta, updated B
    """
    B = mu * B + grad
    X = newton_schulz5(B, steps=ns_steps, eps=eps)
    M, N = theta.shape
    norm_B = np.linalg.norm(B, ord='fro')
    scale = np.sqrt(M * N) / (norm_B + eps)
    theta_new = theta - eta * scale * X
    return theta_new, B
