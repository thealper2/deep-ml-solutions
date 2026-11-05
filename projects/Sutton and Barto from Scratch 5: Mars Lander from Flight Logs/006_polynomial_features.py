import itertools
import numpy as np


def polynomial_features(states, order=1, scale=None):
    X = np.asarray(states, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    h = X[:, 0]
    v = X[:, 1]
    f = X[:, 2]
    exps = list(itertools.product(range(order + 1), repeat=3))
    cols = [h**c1 * v**c2 * f**c3 for (c1, c2, c3) in exps]
    out = np.stack(cols, axis=1)
    if scale is not None:
        out = out / np.asarray(scale, dtype=float)

    return out