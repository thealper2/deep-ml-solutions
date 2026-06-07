import numpy as np

def newtonschulz5(G: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    """
    Apply the Newton-Schulz (quintic) iteration for 5 steps to matrix G.
    Args:
        G: 2D NumPy array
        steps: Number of iteration steps (default 5)
        eps: Small constant for stability
    Returns:
        Matrix after Newton-Schulz iteration
    """
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

def muon_update(theta: np.ndarray, grad: np.array, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    """
    Performs one Muon optimizer update (Algorithm 2). Returns the updated parameter, new B, and the preconditioned update.
    Args:
        theta: Parameter matrix (2D NumP array)
        grad: Gradient matrix (same shape)
        B_prev: Previous B matrix (momentum)
        mu: Momentum factor (0 <= mu < 1)
        lr: Learning rate (step size)
    Returns:
        theta_new: Updated parameter matrix
        B_new: Updated B matrix
        O: Preconditioned update
    """
    B_new = mu * B_prev + grad
    O = newtonschulz5(B_new)
    theta_new = theta - lr * O
    return theta_new, B_new, O
