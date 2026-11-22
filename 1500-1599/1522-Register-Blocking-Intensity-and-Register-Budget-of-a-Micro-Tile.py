import math


def intensity(tm, tn):
    return 2 * tm * tn / (tm + tn)


def registers(tm, tn):
    return tm * tn + tm + tn


def loads_per_output(tm, tn, K):
    return K * (tm + tn) / (tm * tn)


def best_tile(budget, max_side=16):
    best = None
    for tm in range(1, max_side + 1):
        for tn in range(1, max_side + 1):
            r = registers(tm, tn)
            if r > budget:
                continue
            key = (-intensity(tm, tn), r, tm)
            if best is None or key < best[0]:
                best = (key, (tm, tn))

    return best[1]


def cache_block(cache_bytes, elem_bytes=4):
    return math.isqrt(cache_bytes // (3 * elem_bytes))
