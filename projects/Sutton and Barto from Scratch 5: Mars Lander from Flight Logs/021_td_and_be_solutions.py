import numpy as np


def td_and_be_solutions(X, mu, P, R, gamma=1.0):
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)

    n = X.shape[0]
    r_bar = np.sum(P * R, axis=1)
    PX = gamma * (P[:, :n] @ X)
    D = np.diag(mu)

    A_td = X.T @ D @ (X - PX)
    b_td = X.T @ D @ r_bar
    w_td, *_ = np.linalg.lstsq(A_td, b_td, rcond=None)

    sqrt_mu = np.sqrt(mu)
    A_be = sqrt_mu[:, None] * (PX - X)
    b_be = sqrt_mu * r_bar
    w_be, *_ = np.linalg.lstsq(A_be, -b_be, rcond=None)

    return w_td, w_be