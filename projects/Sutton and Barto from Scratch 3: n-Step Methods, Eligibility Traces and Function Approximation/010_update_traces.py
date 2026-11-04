import numpy as np

def update_traces(traces, state, lam, gamma, mode):
    out = np.array(traces, dtype=float, copy=True)
    out *= gamma * lam

    if mode == "accumulating":
        out[state] += 1.0
    elif mode == "replacing":
        out[state] = 1.0
    else:
        raise ValueError(f"unknown trace mode: {mode!r}")

    return out