import numpy as np


def dyna_q_plus_planning(q_table, model, n_planning, time, kappa, alpha, gamma, rng):
    q = np.array(q_table, dtype=float, copy=True)
    if not model or n_planning <= 0:
        return q

    keys = list(model.keys())
    indices = rng.integers(0, len(keys), size=n_planning)
    for i in indices:
        state, action = keys[i]
        next_state, reward, last_time = model[(state, action)]
        bonus = kappa * np.sqrt(time - last_time)
        q = q_learning_update(q, state, action, reward + bonus, next_state, alpha, gamma)

    return q