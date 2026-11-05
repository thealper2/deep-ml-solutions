import numpy as np


def tsitsiklis_van_roy(w0, gamma, epsilon, n_iters):
    x = np.array([1.0, 2.0])
    P = np.array([[0.0, 1.0], [0.0, 1.0 - epsilon]])
    w = float(w0)
    ws = [w]
    for _ in range(n_iters):
        targets = gamma * (P @ (x * w))
        A = x.reshape(-1, 1)
        sol, *_ = np.linalg.lstsq(A, targets, rcond=None)
        w = float(sol[0])
        ws.append(w)

    return np.array(ws, dtype=float)