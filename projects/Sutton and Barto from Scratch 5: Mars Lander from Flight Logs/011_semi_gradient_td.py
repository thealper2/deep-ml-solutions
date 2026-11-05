import numpy as np


def semi_gradient_td(X, X_next, rewards, dones, alpha, rho=None, gamma=1.0):
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

    w = np.zeros(d, dtype=float)
    for t in range(T):
        x = X[t]
        xn = X_next[t]
        delta = rewards[t] + gamma * float(w @ xn) - float(w @ x)
        w = w + alpha * rho[t] * delta * x

    return w