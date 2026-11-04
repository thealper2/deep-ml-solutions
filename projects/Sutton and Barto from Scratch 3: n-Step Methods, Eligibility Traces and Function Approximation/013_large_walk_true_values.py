import numpy as np

def large_walk_true_values(n_states, max_jump):
    P = np.zeros((n_states, n_states), dtype=float)
    r = np.zeros(n_states, dtype=float)

    for s in range(n_states):
        for j in range(1, max_jump + 1):
            p = 0.5 / max_jump
            left = s - j
            right = s + j

            if left < 0:
                r[s] += p * (-1.0)
            else:
                P[s, left] += p

            if right >= n_states:
                r[s] += p * (1.0)
            else:
                P[s, right] += p

    A = np.eye(n_states) - P
    v = np.linalg.solve(A, r)
    return v