import numpy as np


def choose_threshold(y_true, p, cost_fp, cost_fn):
    """(threshold, cost) with the lowest fp * cost_fp + fn * cost_fn over 0.05..0.95."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    best_t = None
    best_cost = None
    for i in range(1, 20):
        t = round(i * 0.05, 2)
        pred = (p >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp * cost_fp + fn * cost_fn
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_t = t

    return (best_t, best_cost)