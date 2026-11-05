import numpy as np


def importance_ratios(logs):
    states = logs["states"]
    actions = logs["actions"]
    dones = np.asarray(logs["dones"], dtype=bool)
    T = len(actions)

    rho = np.zeros(T, dtype=float)
    for t in range(T):
        s = tuple(int(x) for x in states[t])
        a = int(actions[t])
        pi = autopilot_probs(s, "fast")[a]
        b = autopilot_probs(s, "cautious")[a]
        rho[t] = pi / b

    rho_to_end = np.zeros(T, dtype=float)
    running = 1.0
    for t in range(T - 1, -1, -1):
        running = rho[t] * running
        rho_to_end[t] = running
        if t == 0 or dones[t - 1]:
            running = 1.0

    return rho, rho_to_end