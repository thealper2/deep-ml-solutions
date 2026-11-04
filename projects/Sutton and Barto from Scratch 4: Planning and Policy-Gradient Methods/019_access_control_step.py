import numpy as np


def access_control_step(state, action, rng, n_servers=10, free_prob=0.06):
    free, prio = state
    priorities = (1.0, 2.0, 4.0, 8.0)

    reward = 0.0
    if action == 1 and free > 0:
        reward = priorities[prio]
        free -= 1

    busy = n_servers - free
    freed = int(rng.binomial(busy, free_prob))
    free += freed

    next_prio = int(rng.integers(0, 4))
    return ((int(free), next_prio), float(reward))