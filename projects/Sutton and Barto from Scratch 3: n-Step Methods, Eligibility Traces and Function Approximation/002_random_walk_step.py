import numpy as np

def random_walk_step(state, n_states, rng, max_jump=1):
    left = rng.random() < 0.5
    if max_jump == 1:
        jump = 1
    else:
        jump = int(rng.integers(1, max_jump + 1))

    nxt = state - jump if left else state + jump

    if nxt < 0:
        return (-1, -1.0, True)
    if nxt >= n_states:
        return (n_states, 1.0, True)
    return (int(nxt), 0.0, False)