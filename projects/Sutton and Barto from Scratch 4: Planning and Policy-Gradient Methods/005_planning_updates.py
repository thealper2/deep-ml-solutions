import numpy as np


def planning_updates(q_table, model, n_planning, alpha, gamma, rng):
    q = np.array(q_table, dtype=float, copy=True)
    if not model or n_planning <= 0:
        return q

    keys = list(model.keys())
    indices = rng.integers(0, len(keys), size=n_planning)
    for i in indices:
        state, action = keys[i]
        next_state, reward = model[(state, action)]
        q = q_learning_update(q, state, action, reward, next_state, alpha, gamma)

    return q