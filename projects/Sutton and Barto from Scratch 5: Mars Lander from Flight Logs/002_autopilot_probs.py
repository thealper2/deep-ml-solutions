import numpy as np


def autopilot_probs(state, kind):
    h, v, f = state

    if kind == "cautious":
        target = -1
        eps = 0.2
    elif kind == "fast":
        target = -3 if h >= 9 else -1
        eps = 0.15
    else:
        raise ValueError(f"unknown autopilot kind: {kind}")

    if v < target:
        rule = 2
    elif v == target:
        rule = 1
    else:
        rule = 0

    probs = np.full(3, eps / 3.0, dtype=float)
    probs[rule] += 1.0 - eps
    return probs