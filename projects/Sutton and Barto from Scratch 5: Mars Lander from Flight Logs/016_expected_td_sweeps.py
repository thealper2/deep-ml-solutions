import numpy as np


def expected_td_sweeps(X, X_next, rewards, dones, alpha, n_sweeps, rho=None, emphasis=None, gamma=1.0, w0=None):
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
    if emphasis is None:
        emphasis = np.ones(T, dtype=float)
    else:
        emphasis = np.asarray(emphasis, dtype=float)

    c = rho * emphasis

    if w0 is None:
        w = np.zeros(d, dtype=float)
    else:
        w = np.array(w0, dtype=float, copy=True)

    norms = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_sweeps):
            delta = rewards + gamma * (X_next @ w) - (X @ w)
            w = w + alpha * ((c * delta) @ X) / T
            norms.append(float(np.linalg.norm(w)))

    return w, norms