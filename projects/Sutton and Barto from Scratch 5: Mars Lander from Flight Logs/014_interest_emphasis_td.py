import numpy as np


def interest_emphasis_td(n, alpha, n_episodes, use_emphasis):
    groups = [0, 0, 1, 1]
    interest = [1.0, 0.0, 0.0, 0.0]
    w = np.zeros(2, dtype=float)

    for _ in range(n_episodes):
        M = np.zeros(4, dtype=float)
        for t in range(4):
            steps = min(n, 4 - t)
            G = float(steps)
            if t + n < 4:
                G += w[groups[t + n]]
            M[t] = interest[t] + (M[t - n] if t - n >= 0 else 0.0)
            weight = M[t] if use_emphasis else 1.0
            g = groups[t]
            w[g] += alpha * weight * (G - w[g])

    return w