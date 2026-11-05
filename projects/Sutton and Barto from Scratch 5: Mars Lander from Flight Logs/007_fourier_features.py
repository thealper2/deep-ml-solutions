import itertools
import numpy as np


def unit_states(states):
    S = np.asarray(states, dtype=float)
    if S.ndim == 1:
        S = S.reshape(1, -1)

    h = S[:, 0] / MAX_ALTITUDE
    v = (S[:, 1] + MAX_SPEED) / (2 * MAX_SPEED)
    f = S[:, 2] / MAX_FUEL
    return np.stack([h, v, f], axis=1)

def fourier_features(states, order):
    S = unit_states(states)
    cs = list(itertools.product(range(order + 1), repeat=3))
    C = np.array(cs, dtype=float)
    return np.cos(np.pi * S @ C.T)