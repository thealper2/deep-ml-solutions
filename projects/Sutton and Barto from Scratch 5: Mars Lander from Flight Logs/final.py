"""
Sutton and Barto from Scratch 5: Mars Lander from Flight Logs — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  lander_step ──
import numpy as np

MAX_ALTITUDE = 16
MAX_SPEED = 5
MAX_FUEL = 24
SAFE_SPEED = 2
START = (16, 0, 24)


def lander_step(state, action, rng):
    h, v, f = state
    thrust = min(action, f)
    new_fuel = f - thrust

    u = rng.random()
    if u < 0.1:
        gust = -1
    elif u >= 0.9:
        gust = 1
    else:
        gust = 0

    new_v = v + thrust - 1 + gust
    new_v = max(-MAX_SPEED, min(MAX_SPEED, new_v))
    new_v = int(new_v)

    new_h = h + new_v

    if new_h <= 0:
        reward = 1.0 if new_v >= -SAFE_SPEED else 0.0
        return ((0, new_v, int(new_fuel)), reward, True)

    return ((int(min(new_h, MAX_ALTITUDE)), new_v, int(new_fuel)), 0.0, False)

# ── Step 002  autopilot_probs ──
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

# ── Step 003  fly ──
import numpy as np


def fly(kind, n_flights, rng, start=START):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    flights = []

    for fidx in range(n_flights):
        state = tuple(int(x) for x in start)
        while True:
            probs = autopilot_probs(state, kind)
            cdf = np.cumsum(probs)
            action = min(int(np.searchsorted(cdf, rng.random(), side="right")), 2)
            next_state, reward, done = lander_step(state, action, rng)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            flights.append(fidx)

            state = next_state
            if done:
                break

    states = np.array(states, dtype=int)
    actions = np.array(actions, dtype=int)
    rewards = np.array(rewards, dtype=float)
    next_states = np.array(next_states, dtype=int)
    dones = np.array(dones, dtype=bool)
    flights = np.array(flights, dtype=int)

    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + running
        returns[t] = running
        if t == 0 or dones[t - 1]:
            running = 0.0

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "flight": flights,
        "returns": returns,
    }

# ── Step 004  landing_rate ──
import numpy as np


def landing_rate(kind, n_flights, rng, start=START):
    logs = fly(kind, n_flights, rng, start)
    return float(logs["rewards"][logs["dones"]].mean())

# ── Step 005  exact_values ──
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

# ── Step 006  polynomial_features ──
import itertools
import numpy as np


def polynomial_features(states, order=1, scale=None):
    X = np.asarray(states, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    h = X[:, 0]
    v = X[:, 1]
    f = X[:, 2]
    exps = list(itertools.product(range(order + 1), repeat=3))
    cols = [h**c1 * v**c2 * f**c3 for (c1, c2, c3) in exps]
    out = np.stack(cols, axis=1)
    if scale is not None:
        out = out / np.asarray(scale, dtype=float)

    return out

# ── Step 007  fourier_features ──
import itertools
import numpy as np


def unit_states(states):
    S = np.asarray(states, dtype=float)
    if S.ndim == 1:
        S = S.reshape(1, -1)

    h = S[:, 0] / MAX_ALTITUDE
    v = (S[:, 1] + MAX_SPEED) / (2 * MAX_SPEED)
    f = S[:, 2] / MAX_FUEL
    return np.stack([h, v, f], axis=1)

def fourier_features(states, order):
    S = unit_states(states)
    cs = list(itertools.product(range(order + 1), repeat=3))
    C = np.array(cs, dtype=float)
    return np.cos(np.pi * S @ C.T)

# ── Step 008  rbf_features ──
import numpy as np


def rbf_features(states, centers, sigma):
    S = unit_states(states)
    C = np.asarray(centers, dtype=float).reshape(-1, 3)
    d2 = np.sum((S[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * sigma ** 2))

# ── Step 009  tile_features ──
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

# ── Step 010  step_size_rule ──
import numpy as np


def step_size_rule(X, tau):
    X = np.asarray(X, dtype=float)
    mean_sq = np.mean(np.sum(X * X, axis=1))
    return float(1.0 / (tau * mean_sq))

# ── Step 011  semi_gradient_td ──
import numpy as np


def semi_gradient_td(X, X_next, rewards, dones, alpha, rho=None, gamma=1.0):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    if rho is None:
        rho = np.ones(T, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)

    w = np.zeros(d, dtype=float)
    for t in range(T):
        x = X[t]
        xn = X_next[t]
        delta = rewards[t] + gamma * float(w @ xn) - float(w @ x)
        w = w + alpha * rho[t] * delta * x

    return w

# ── Step 012  lstd ──
import numpy as np


def lstd(X, X_next, rewards, dones, epsilon=1e-3, rho=None, gamma=1.0):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    if rho is None:
        rho = np.ones(T, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)

    A_inv = np.eye(d) / epsilon
    b = np.zeros(d, dtype=float)

    for t in range(T):
        x = X[t]
        xn = X_next[t]
        u = rho[t] * (x - gamma * xn)
        v = A_inv.T @ u
        A_inv = A_inv - np.outer(A_inv @ x, v) / (1.0 + v @ x)
        b = b + rho[t] * rewards[t] * x

    return A_inv @ b

# ── Step 013  kernel_value ──
import numpy as np


def kernel_value(query_states, memory_states, memory_targets, bandwidth):
    Q = unit_states(query_states)
    M = unit_states(memory_states)
    g = np.asarray(memory_targets, dtype=float)

    d2 = np.sum((Q[:, None, :] - M[None, :, :]) ** 2, axis=2)
    d2 = d2 - d2.min(axis=1, keepdims=True)
    k = np.exp(-d2 / (2.0 * bandwidth ** 2))
    return (k @ g) / k.sum(axis=1)

# ── Step 014  interest_emphasis_td ──
import numpy as np


def interest_emphasis_td(n, alpha, n_episodes, use_emphasis):
    groups = [0, 0, 1, 1]
    interest = [1.0, 0.0, 0.0, 0.0]
    w = np.zeros(2, dtype=float)

    for _ in range(n_episodes):
        M = np.zeros(4, dtype=float)
        for t in range(4):
            steps = min(n, 4 - t)
            G = float(steps)
            if t + n < 4:
                G += w[groups[t + n]]
            M[t] = interest[t] + (M[t - n] if t - n >= 0 else 0.0)
            weight = M[t] if use_emphasis else 1.0
            g = groups[t]
            w[g] += alpha * weight * (G - w[g])

    return w

# ── Step 015  importance_ratios ──
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

# ── Step 016  expected_td_sweeps ──
import numpy as np


def expected_td_sweeps(X, X_next, rewards, dones, alpha, n_sweeps, rho=None, emphasis=None, gamma=1.0, w0=None):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    if rho is None:
        rho = np.ones(T, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)
    if emphasis is None:
        emphasis = np.ones(T, dtype=float)
    else:
        emphasis = np.asarray(emphasis, dtype=float)

    c = rho * emphasis

    if w0 is None:
        w = np.zeros(d, dtype=float)
    else:
        w = np.array(w0, dtype=float, copy=True)

    norms = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_sweeps):
            delta = rewards + gamma * (X_next @ w) - (X @ w)
            w = w + alpha * ((c * delta) @ X) / T
            norms.append(float(np.linalg.norm(w)))

    return w, norms

# ── Step 017  deadly_triad_postmortem ──
import numpy as np


def deadly_triad_postmortem(logs_b, logs_pi, features, aggregate, tau, n_sweeps):
    rho, rho_to_end = importance_ratios(logs_b)

    def run(X, X_next, rewards, dones, c):
        Xs = X * np.sqrt(c)[:, None]
        alpha = step_size_rule(Xs, tau)
        _, norms = expected_td_sweeps(X, X_next, rewards, dones, alpha, n_sweeps, rho=c)
        return norms[-1]

    Xb = features(logs_b["states"])
    Xnb = features(logs_b["next_states"])
    res = {}
    res["all three legs"] = run(Xb, Xnb, logs_b["rewards"], logs_b["dones"], rho)

    Xp = features(logs_pi["states"])
    Xnp = features(logs_pi["next_states"])
    ones_p = np.ones(len(logs_pi["actions"]), dtype=float)
    res["on-policy data"] = run(Xp, Xnp, logs_pi["rewards"], logs_pi["dones"], ones_p)

    Ab = aggregate(logs_b["states"])
    Anb = aggregate(logs_b["next_states"])
    res["state aggregation"] = run(Ab, Anb, logs_b["rewards"], logs_b["dones"], rho)

    zeros_next = np.zeros_like(Xb)
    all_done = np.ones(len(logs_b["actions"]), dtype=bool)
    res["Monte Carlo targets"] = run(Xb, zeros_next, logs_b["returns"], all_done, rho_to_end)

    return res

# ── Step 018  tsitsiklis_van_roy ──
import numpy as np


def tsitsiklis_van_roy(w0, gamma, epsilon, n_iters):
    x = np.array([1.0, 2.0])
    P = np.array([[0.0, 1.0], [0.0, 1.0 - epsilon]])
    w = float(w0)
    ws = [w]
    for _ in range(n_iters):
        targets = gamma * (P @ (x * w))
        A = x.reshape(-1, 1)
        sol, *_ = np.linalg.lstsq(A, targets, rcond=None)
        w = float(sol[0])
        ws.append(w)

    return np.array(ws, dtype=float)

# ── Step 019  baird_counterexample ──
import numpy as np


def baird_features():
    X = np.zeros((7, 8), dtype=float)
    for i in range(6):
        X[i, i] = 2.0
        X[i, 7] = 1.0
    X[6, 6] = 1.0
    X[6, 7] = 2.0
    return X


def baird_counterexample(n_updates, alpha, rng=None, gamma=0.99):
    X = baird_features()
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    history = [w.copy()]

    if rng is None:
        x6 = X[6]
        for _ in range(n_updates):
            delta = gamma * (x6 @ w) - (X @ w)
            w = w + alpha / 7.0 * (X.T @ delta)
            history.append(w.copy())
    else:
        state = int(rng.random() * 7)
        for _ in range(n_updates):
            u = rng.random()
            if u < 6.0 / 7.0:
                next_state = int(u * 7)
                rho = 0.0
            else:
                next_state = 6
                rho = 7.0
            delta = 0.0 + gamma * (X[next_state] @ w) - (X[state] @ w)
            w = w + alpha * rho * delta * X[state]
            history.append(w.copy())
            state = next_state

    return np.array(history, dtype=float)

# ── Step 020  value_error_objectives ──
import numpy as np


def value_error_objectives(X, w, mu, P, R, v_true, gamma=1.0):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    v_true = np.asarray(v_true, dtype=float)

    n = X.shape[0]
    v = X @ w

    v_aug = np.concatenate([v, [0.0]])
    delta = R + gamma * v_aug[None, :] - v[:, None]

    bellman = np.sum(P * delta, axis=1)
    VE = float(np.sum(mu * (v - v_true) ** 2))
    BE = float(np.sum(mu * bellman ** 2))

    D = np.diag(mu)
    A = X.T @ D @ X
    Pi = X @ np.linalg.pinv(A) @ X.T @ D
    PBE = float(np.sum(mu * (Pi @ bellman) ** 2))

    TDE = float(np.sum(mu * np.sum(P * delta ** 2, axis=1)))

    return {"VE": VE, "BE": BE, "PBE": PBE, "TDE": TDE}

# ── Step 021  td_and_be_solutions ──
import numpy as np


def td_and_be_solutions(X, mu, P, R, gamma=1.0):
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)

    n = X.shape[0]
    r_bar = np.sum(P * R, axis=1)
    PX = gamma * (P[:, :n] @ X)
    D = np.diag(mu)

    A_td = X.T @ D @ (X - PX)
    b_td = X.T @ D @ r_bar
    w_td, *_ = np.linalg.lstsq(A_td, b_td, rcond=None)

    sqrt_mu = np.sqrt(mu)
    A_be = sqrt_mu[:, None] * (PX - X)
    b_be = sqrt_mu * r_bar
    w_be, *_ = np.linalg.lstsq(A_be, -b_be, rcond=None)

    return w_td, w_be

# ── Step 022  expected_tdc_sweeps ──
import numpy as np


def expected_tdc_sweeps(X, X_next, rewards, dones, rho, alpha, beta, n_sweeps, gamma=1.0, gtd2=False, w0=None):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    rho = np.asarray(rho, dtype=float)

    if w0 is None:
        w = np.zeros(d, dtype=float)
    else:
        w = np.array(w0, dtype=float, copy=True)
    v = np.zeros(d, dtype=float)

    for _ in range(n_sweeps):
        delta = rewards + gamma * (X_next @ w) - (X @ w)
        xv = X @ v

        if gtd2:
            w = w + alpha * (rho[:, None] * (X - gamma * X_next) * xv[:, None]).sum(axis=0) / T
        else:
            w = w + alpha * (rho[:, None] * (delta[:, None] * X - gamma * X_next * xv[:, None])).sum(axis=0) / T

        v = v + beta * (rho[:, None] * (delta - xv)[:, None] * X).sum(axis=0) / T

    return w, v

# ── Step 023  follow_on_emphasis ──
import numpy as np


def follow_on_emphasis(rho, dones, gamma=1.0, interest=None):
    rho = np.asarray(rho, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T = len(rho)

    if interest is None:
        interest = np.ones(T, dtype=float)
    else:
        interest = np.asarray(interest, dtype=float)

    M = np.zeros(T, dtype=float)
    carried = 0.0
    for t in range(T):
        if t == 0 or dones[t - 1]:
            carried = 0.0
        M[t] = carried + interest[t]
        carried = gamma * rho[t] * M[t]

    return M

# ── Step 024  td_lambda ──
import numpy as np


def td_lambda(X, X_next, rewards, dones, alpha, lam, true_online=False, gamma=1.0):
    X = np.asarray(X, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T, d = X.shape

    X_next = X_next.copy()
    X_next[dones] = 0.0

    w = np.zeros(d, dtype=float)
    z = np.zeros(d, dtype=float)
    v_old = 0.0

    for t in range(T):
        x = X[t]
        xn = X_next[t]
        V = float(w @ x)
        V_next = float(w @ xn)
        delta = rewards[t] + gamma * V_next - V

        if true_online:
            z = gamma * lam * z + (1.0 - alpha * gamma * lam * float(z @ x)) * x
            w = w + alpha * (delta + V - v_old) * z - alpha * (V - v_old) * x
            v_old = V_next
        else:
            z = gamma * lam * z + x
            w = w + alpha * delta * z

        if dones[t]:
            z = np.zeros(d, dtype=float)
            v_old = 0.0

    return w

# ── Step 025  go_no_go_table ──
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

# ── Scaffold (runner) ──
"""Sutton & Barto from scratch 5: Mars Lander from Flight Logs.
Feature construction and LSTD (chapter 9), off-policy divergence and its fixes (chapter 11), true online TD(lambda) (chapter 12)."""
import itertools
import numpy as np


def sqrt_ve(X, w, truth):
    return float(np.sqrt(np.mean((X @ w - truth) ** 2)))


def main():
    # --- Part 1: the lander, its two autopilots, the flight logs and the exact values ---
    logs = fly("cautious", 500, np.random.default_rng(1))
    D = logs["dones"]
    print("Flight logs from the cautious autopilot:", int(D.sum()), "flights,", len(D), "transitions")
    print("Landed safely in the logs:", round(float(logs["rewards"][D].mean()), 3))
    pi_logs = fly("fast", 500, np.random.default_rng(4))
    fuel_b = float(logs["next_states"][D][:, 2].mean())
    fuel_pi = float(pi_logs["next_states"][pi_logs["dones"]][:, 2].mean())
    print("Mean fuel left at touchdown: cautious", round(fuel_b, 1), "| fast", round(fuel_pi, 1))
    V_b, V_pi = exact_values("cautious"), exact_values("fast")
    h0, v0, f0 = START
    print("Exact landing probability from the start (dynamic programming): cautious",
          round(float(V_b[h0, v0 + MAX_SPEED, f0]), 4), "| fast", round(float(V_pi[h0, v0 + MAX_SPEED, f0]), 4))
    S = logs["states"]
    truth = V_b[S[:, 0], S[:, 1] + MAX_SPEED, S[:, 2]]

    # --- Part 2: features, scored by sqrt(VE) of their LSTD solution over the logged states ---
    raw = polynomial_features(S)
    scale = np.sqrt((raw ** 2).mean(axis=0))
    grid = np.array(list(itertools.product(np.linspace(0, 1, 4), repeat=3)))
    families = {
        "polynomial, order 1 (raw telemetry / RMS)": lambda s: polynomial_features(s, 1, scale),
        "Fourier, order 2": lambda s: fourier_features(s, 2),
        "Fourier, order 3": lambda s: fourier_features(s, 3),
        "radial basis, 4x4x4 grid, sigma 0.2": lambda s: rbf_features(s, grid, 0.2),
        "tile coding, 4 tilings of 3x3x3": lambda s: tile_features(s, 4, 3),
    }
    print("On-policy prediction of the cautious autopilot (LSTD on all 500 flights):")
    for name, feat in families.items():
        X = feat(S)
        w = lstd(X, feat(logs["next_states"]), logs["rewards"], D)
        print(f"  {name:42s} d={X.shape[1]:3d}  alpha rule (tau=10)={step_size_rule(X, 10):.4f}  sqrt(VE)={sqrt_ve(X, w, truth):.4f}")
    sample = np.arange(0, len(S), 20)
    kv = kernel_value(S[sample], S, logs["returns"], 0.06)
    print(f"  {'kernel regression on returns, bandwidth 0.06':42s}                              sqrt(VE)={float(np.sqrt(np.mean((kv - truth[sample]) ** 2))):.4f}")

    # --- Part 3: data efficiency (LSTD vs semi-gradient TD) and interest/emphasis (Example 9.4) ---
    feat = families["Fourier, order 3"]
    X_all = feat(S)
    for k in (10, 50, 500):
        m = logs["flight"] < k
        X, X_next = feat(S[m]), feat(logs["next_states"][m])
        w_l = lstd(X, X_next, logs["rewards"][m], D[m])
        w_t = semi_gradient_td(X, X_next, logs["rewards"][m], D[m], step_size_rule(X, 10))
        print(f"After {k:3d} flights: LSTD sqrt(VE)={sqrt_ve(X_all, w_l, truth):.4f}   semi-gradient TD(0) sqrt(VE)={sqrt_ve(X_all, w_t, truth):.4f}")
    print("Example 9.4, two-step TD: without emphasis", np.round(interest_emphasis_td(2, 0.05, 2000, False), 3).tolist(),
          "| with emphasis", np.round(interest_emphasis_td(2, 0.05, 2000, True), 3).tolist())

    # --- Part 4: the incident. Evaluate the fast autopilot from the cautious logs ---
    rho, rho_to_end = importance_ratios(logs)
    print("Importance ratios pi/b on the logs: max", round(float(rho.max()), 2), "| mean", round(float(rho.mean()), 3),
          "| share below 0.1:", round(float((rho < 0.1).mean()), 3))
    poly = families["polynomial, order 1 (raw telemetry / RMS)"]
    X, X_next = poly(S), poly(logs["next_states"])
    alpha = step_size_rule(X * np.sqrt(rho)[:, None], 1)
    w_semi, norms = expected_td_sweeps(X, X_next, logs["rewards"], D, alpha, 600, rho=rho)
    print("Off-policy semi-gradient TD on the logs, |w| after sweeps 1, 150, 300, 450, 600:",
          [f"{norms[i]:.2e}" for i in (0, 149, 299, 449, 599)])
    post = deadly_triad_postmortem(logs, pi_logs, poly, lambda s: tile_features(s, 1, 3), 1, 600)
    print("Post-mortem, |w| after 600 sweeps:")
    for name, norm in post.items():
        print(f"  {name:22s} {norm:.3e}")
    print("Tsitsiklis-Van Roy, least-squares DP w_k (k = 0..5): gamma 0.9", np.round(tsitsiklis_van_roy(1.0, 0.9, 0.0, 5), 3).tolist(),
          "| gamma 0.8", np.round(tsitsiklis_van_roy(1.0, 0.8, 0.0, 5), 3).tolist())
    td_hist = baird_counterexample(1000, 0.01, np.random.default_rng(6))
    dp_hist = baird_counterexample(1000, 0.01)
    print("Baird, semi-gradient off-policy TD: w8 at steps 0, 500, 1000 =", np.round(td_hist[[0, 500, 1000], 7], 1).tolist())
    print("Baird, semi-gradient DP (expected update): w8 at sweeps 0, 500, 1000 =", np.round(dp_hist[[0, 500, 1000], 7], 1).tolist())

    # --- Part 5: the geometry of the objectives, the fixes, and the go/no-go call ---
    Xs = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P = np.zeros((4, 5)); P[0, 2] = P[1, 3] = P[2, 4] = P[3, 4] = 1.0
    R = np.zeros((4, 5)); R[2, 4] = 1.0
    mu, v_true = np.full(4, 0.25), np.array([1.0, 0.0, 1.0, 0.0])
    w_td, w_be = td_and_be_solutions(Xs, mu, P, R)
    print("A-presplit: TD fixed point (A, B, C) =", (np.round(w_td, 3) + 0.0).tolist(), "| BE minimiser =", (np.round(w_be, 3) + 0.0).tolist())
    for label, wv in (("TD fixed point", w_td), ("BE minimiser", w_be)):
        obj = value_error_objectives(Xs, wv, mu, P, R, v_true)
        print(f"  {label:15s} VE={obj['VE']:.4f} BE={obj['BE']:.4f} PBE={obj['PBE']:.4f} TDE={obj['TDE']:.4f}")
    Xb, w0 = baird_features(), np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
    Pb = np.zeros((7, 8)); Pb[:, 6] = 1.0
    mub = np.full(7, 1 / 7)
    expected_args = (Xb, np.tile(Xb[6], (7, 1)), np.zeros(7), np.zeros(7, dtype=bool))
    w_tdc_b, _ = expected_tdc_sweeps(*expected_args, np.ones(7), 0.005, 0.05, 1000, gamma=0.99, w0=w0)
    o = value_error_objectives(Xb, w_tdc_b, mub, Pb, np.zeros((7, 8)), np.zeros(7), gamma=0.99)
    print(f"Baird, expected TDC after 1000 sweeps: sqrt(PBE)={np.sqrt(o['PBE']):.4f}  sqrt(VE)={np.sqrt(o['VE']):.3f}")
    m = np.linalg.solve(np.eye(7) - 0.99 * Pb[:, :7].T, mub) * 7
    w_em_b, _ = expected_td_sweeps(*expected_args, 0.03, 1000, emphasis=m, gamma=0.99, w0=w0)
    o = value_error_objectives(Xb, w_em_b, mub, Pb, np.zeros((7, 8)), np.zeros(7), gamma=0.99)
    print(f"Baird, expected emphatic TD after 1000 sweeps: sqrt(VE)={np.sqrt(o['VE']):.4f}")

    x0 = poly([START])[0]
    estimates = {"semi-gradient TD": float(x0 @ w_semi)}
    agg = lambda s: tile_features(s, 1, 3)
    estimates["state aggregation (LSTD)"] = float(agg([START])[0] @ lstd(agg(S), agg(logs["next_states"]), logs["rewards"], D, rho=rho))
    starts = np.flatnonzero(np.r_[True, D[:-1]])
    weights = rho_to_end[starts]
    estimates["Monte Carlo, ordinary IS"] = float(np.mean(weights * logs["returns"][starts]))
    estimates["Monte Carlo, weighted IS"] = float(np.sum(weights * logs["returns"][starts]) / np.sum(weights))
    estimates["off-policy LSTD"] = float(x0 @ lstd(X, X_next, logs["rewards"], D, rho=rho))
    w_tdc, _ = expected_tdc_sweeps(X, X_next, logs["rewards"], D, rho, 0.1, 0.1, 10000)
    estimates["TDC, 10000 sweeps"] = float(x0 @ w_tdc)
    w_gtd2, _ = expected_tdc_sweeps(X, X_next, logs["rewards"], D, rho, 0.1, 0.1, 10000, gtd2=True)
    estimates["GTD2, 10000 sweeps"] = float(x0 @ w_gtd2)
    M = follow_on_emphasis(rho, D)
    w_em, _ = expected_td_sweeps(X, X_next, logs["rewards"], D, step_size_rule(X * np.sqrt(M * rho)[:, None], 1), 3000,
                                 rho=rho, emphasis=M)
    estimates["emphatic TD"] = float(x0 @ w_em)
    truth_pi = landing_rate("fast", 4000, np.random.default_rng(3))
    print("Go/no-go: switch to the fast autopilot only if its landing rate is at least 0.90")
    for name, value, call, ok in go_no_go_table(estimates, truth_pi, 0.90):
        shown = "     --" if value is None else f"{value:7.4f}"
        print(f"  {name:26s} {shown}  {call:8s} {'agrees with the truth' if ok else 'WRONG'}")
    print(f"  {'truth: fly it 4000 times':26s} {truth_pi:7.4f}  {'GO' if truth_pi >= 0.9 else 'NO-GO'}"
          f"   (exact value {float(V_pi[h0, v0 + MAX_SPEED, f0]):.4f})")

    # --- true online TD(lambda) vs accumulating TD(lambda), on-policy on the cautious logs (figure 12.8's lesson) ---
    tiles = lambda s: tile_features(s, 8, 4)
    Xt, Xt_next = tiles(S), tiles(logs["next_states"])
    for a in (0.5, 1.0, 1.5):
        row = []
        for true_online in (False, True):
            wl = td_lambda(Xt, Xt_next, logs["rewards"], D, a / 8, 0.9, true_online)
            err = sqrt_ve(Xt, wl, truth)
            row.append("diverged" if not np.isfinite(err) or err > 10 else f"{err:.4f}")
        print(f"TD(lambda=0.9) with alpha*|x|^2={a}: accumulating sqrt(VE)={row[0]:>8s}   true online sqrt(VE)={row[1]}")


if __name__ == "__main__":
    main()
