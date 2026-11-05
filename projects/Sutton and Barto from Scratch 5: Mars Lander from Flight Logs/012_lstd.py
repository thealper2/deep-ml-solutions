import numpy as np


def lstd(X, X_next, rewards, dones, epsilon=1e-3, rho=None, gamma=1.0):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    if rho is None:
        rho = np.ones(T, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)

    A_inv = np.eye(d) / epsilon
    b = np.zeros(d, dtype=float)

    for t in range(T):
        x = X[t]
        xn = X_next[t]
        u = rho[t] * (x - gamma * xn)
        v = A_inv.T @ u
        A_inv = A_inv - np.outer(A_inv @ x, v) / (1.0 + v @ x)
        b = b + rho[t] * rewards[t] * x

    return A_inv @ b