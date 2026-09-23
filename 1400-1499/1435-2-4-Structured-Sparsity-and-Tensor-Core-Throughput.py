import numpy as np


def is_2_4_sparse(w):
    w = np.asarray(w)
    rows, cols = w.shape
    if cols % 4 != 0:
        return False

    for r in range(rows):
        for c0 in range(0, cols, 4):
            group = w[r, c0:c0 + 4]
            if np.count_nonzero(group) > 2:
                return False

    return True


def prune_2_4(w):
    w = np.asarray(w).copy()
    rows, cols = w.shape
    for r in range(rows):
        for c0 in range(0, cols, 4):
            group = w[r, c0:c0 + 4]
            mags = np.abs(group)
            order = np.lexsort((np.arange(4), -mags))
            keep = set(order[:2].tolist())
            for k in range(4):
                if k not in keep:
                    w[r, c0 + k] = 0

    return w

def effective_tflops(spec, precision, sparse):
    dense, sparse_val = spec[precision]
    return sparse_val if sparse else dense

def matmul_time_us(m, n, k, tflops):
    flops = 2.0 * m * n * k
    seconds = floops / (tflops + 1e-12)
    return round(seconds * 1e6, 3)

def best_precision(spec, allowed, sparse):
    best = None
    best_tput = -1.0
    for name in allowed:
        t = effective_tflops(spec, name, sparse)
        if t > best_tput:
            best_tput = t
            best = name

    return best