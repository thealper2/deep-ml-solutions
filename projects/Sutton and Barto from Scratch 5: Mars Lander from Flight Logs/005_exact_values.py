import numpy as np


class FixedDraw:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def exact_values(kind, tol=1e-12):
    n_h = MAX_ALTITUDE + 1
    n_v = 2 * MAX_SPEED + 1
    n_f = MAX_FUEL + 1
    V = np.zeros((n_h, n_v, n_f), dtype=float)

    us = (0.05, 0.5, 0.95)
    gust_probs = (0.1, 0.8, 0.1)

    def flat(h, v, f):
        return h * (n_v * n_f) + (v + MAX_SPEED) * n_f + f

    transitions = {}
    for h in range(1, n_h):
        for v in range(-MAX_SPEED, MAX_SPEED + 1):
            for f in range(n_f):
                state = (h, v, f)
                probs = autopilot_probs(state, kind)
                entries = []
                for a in range(3):
                    pa = float(probs[a])
                    if pa == 0.0:
                        continue

                    for u, pg in zip(us, gust_probs):
                        ns, r, done = lander_step(state, a, FixedDraw(u))
                        p = pa * pg
                        if done:
                            entries.append((p, r, -1))
                        else:
                            nh, nv, nf = ns
                            entries.append((p, r, flat(nh, nv, nf)))

                transitions[flat(h, v, f)] = entries

    flat_to_coords = []
    for h in range(n_h):
        for v in range(-MAX_SPEED, MAX_SPEED + 1):
            for f in range(n_f):
                flat_to_coords.append((h, v + MAX_SPEED, f))

    while True:
        delta = 0.0
        newV = V.copy()
        for idx, entries in transitions.items():
            total = 0.0
            for p, r, nidx in entries:
                if nidx == -1:
                    total += p * r
                else:
                    nh, nv, nf = flat_to_coords[nidx]
                    total += p * (r + V[nh, nv, nf])
            h, v, f = flat_to_coords[idx]
            newV[h, v, f] = total
            diff = abs(total - V[h, v, f])
            if diff > delta:
                delta = diff
        V = newV
        if delta < tol:
            break

    return V