import numpy as np


def tile_features(states, n_tilings, tiles_per_dim):
    S = unit_states(states)
    n = S.shape[0]
    k = tiles_per_dim
    per_tiling = (k + 1) ** 3
    out = np.zeros((n, n_tilings * per_tiling), dtype=float)
    base_offsets = np.array([1, 3, 5])
    for t in range(n_tilings):
        offset = ((t * base_offsets) % n_tilings) / (n_tilings * k)
        idx = np.minimum(np.floor((S + offset) * k).astype(int), k)
        flat = (idx[:, 0] * (k + 1) + idx[:, 1]) * (k + 1) + idx[:, 2]
        out[np.arange(n), t * per_tiling + flat] = 1.0

    return out