import numpy as np


def expected_tdc_sweeps(X, X_next, rewards, dones, rho, alpha, beta, n_sweeps, gamma=1.0, gtd2=False, w0=None):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    rho = np.asarray(rho, dtype=float)

    if w0 is None:
        w = np.zeros(d, dtype=float)
    else:
        w = np.array(w0, dtype=float, copy=True)
    v = np.zeros(d, dtype=float)

    for _ in range(n_sweeps):
        delta = rewards + gamma * (X_next @ w) - (X @ w)
        xv = X @ v

        if gtd2:
            w = w + alpha * (rho[:, None] * (X - gamma * X_next) * xv[:, None]).sum(axis=0) / T
        else:
            w = w + alpha * (rho[:, None] * (delta[:, None] * X - gamma * X_next * xv[:, None])).sum(axis=0) / T

        v = v + beta * (rho[:, None] * (delta - xv)[:, None] * X).sum(axis=0) / T

    return w, v