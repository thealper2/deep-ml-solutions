import numpy as np

def n_step_return(rewards, states, t, n, values, gamma=1.0):
    T = len(rewards)
    G = 0.0
    discount = 1.0
    for k in range(n):
        idx = t + k
        if idx >= T:
            return float(G)

        G += discount * rewards[idx]
        discount *= gamma

    if t + n < T:
        G += discount * float(values[states[t + n]])

    return float(G)