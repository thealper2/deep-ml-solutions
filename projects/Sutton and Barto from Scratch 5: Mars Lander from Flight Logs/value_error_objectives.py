import numpy as np


def value_error_objectives(X, w, mu, P, R, v_true, gamma=1.0):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    v_true = np.asarray(v_true, dtype=float)

    n = X.shape[0]
    v = X @ w

    v_aug = np.concatenate([v, [0.0]])
    delta = R + gamma * v_aug[None, :] - v[:, None]

    bellman = np.sum(P * delta, axis=1)
    VE = float(np.sum(mu * (v - v_true) ** 2))
    BE = float(np.sum(mu * bellman ** 2))

    D = np.diag(mu)
    A = X.T @ D @ X
    Pi = X @ np.linalg.pinv(A) @ X.T @ D
    PBE = float(np.sum(mu * (Pi @ bellman) ** 2))

    TDE = float(np.sum(mu * np.sum(P * delta ** 2, axis=1)))

    return {"VE": VE, "BE": BE, "PBE": PBE, "TDE": TDE}