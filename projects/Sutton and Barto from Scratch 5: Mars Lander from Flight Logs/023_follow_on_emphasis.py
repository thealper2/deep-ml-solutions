import numpy as np


def follow_on_emphasis(rho, dones, gamma=1.0, interest=None):
    rho = np.asarray(rho, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T = len(rho)

    if interest is None:
        interest = np.ones(T, dtype=float)
    else:
        interest = np.asarray(interest, dtype=float)

    M = np.zeros(T, dtype=float)
    carried = 0.0
    for t in range(T):
        if t == 0 or dones[t - 1]:
            carried = 0.0
        M[t] = carried + interest[t]
        carried = gamma * rho[t] * M[t]

    return M