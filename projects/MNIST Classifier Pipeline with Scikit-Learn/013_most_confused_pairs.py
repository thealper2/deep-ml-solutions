import numpy as np

def most_confused_pairs(cm, k=3):
    off = np.array(cm, dtype=float).copy()
    np.fill_diagonal(off, 0)
    idx = np.argsort(off, axis=None)[::-1][:k]
    result = []
    for flat in idx:
        i, j = np.unravel_index(flat, off.shape)
        result.append((int(i), int(j), round(float(off[i, j]), 3)))

    return result