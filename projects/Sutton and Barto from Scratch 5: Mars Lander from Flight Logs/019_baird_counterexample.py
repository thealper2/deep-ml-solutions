import numpy as np


def baird_features():
    X = np.zeros((7, 8), dtype=float)
    for i in range(6):
        X[i, i] = 2.0
        X[i, 7] = 1.0
    X[6, 6] = 1.0
    X[6, 7] = 2.0
    return X


def baird_counterexample(n_updates, alpha, rng=None, gamma=0.99):
    X = baird_features()
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    history = [w.copy()]

    if rng is None:
        x6 = X[6]
        for _ in range(n_updates):
            delta = gamma * (x6 @ w) - (X @ w)
            w = w + alpha / 7.0 * (X.T @ delta)
            history.append(w.copy())
    else:
        state = int(rng.random() * 7)
        for _ in range(n_updates):
            u = rng.random()
            if u < 6.0 / 7.0:
                next_state = int(u * 7)
                rho = 0.0
            else:
                next_state = 6
                rho = 7.0
            delta = 0.0 + gamma * (X[next_state] @ w) - (X[state] @ w)
            w = w + alpha * rho * delta * X[state]
            history.append(w.copy())
            state = next_state

    return np.array(history, dtype=float)