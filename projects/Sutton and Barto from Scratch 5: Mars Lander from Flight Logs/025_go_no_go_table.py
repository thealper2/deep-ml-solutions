import numpy as np


def go_no_go_table(estimates, truth, threshold):
    true_call = "GO" if truth >= threshold else "NO-GO"
    rows = []
    for name, value in estimates.items():
        if value is None:
            rows.append((name, None, "DIVERGED", False))
            continue
        v = float(value)
        if not np.isfinite(v) or abs(v) > 10:
            rows.append((name, None, "DIVERGED", False))
            continue
        call = "GO" if v >= threshold else "NO-GO"
        rows.append((name, round(v, 4), call, call == true_call))
    return rows