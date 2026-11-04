"""
Sutton and Barto from Scratch 3: n-Step Methods, Eligibility Traces and Function Approximation — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  random_walk_true_values ──
import numpy as np

def random_walk_true_values(n_states):
    i = np.arange(n_states, dtype=float)
    return -1.0 + 2.0 * (i + 1.0) / (n_states + 1.0)

# ── Step 002  random_walk_step ──
import numpy as np

def random_walk_step(state, n_states, rng, max_jump=1):
    left = rng.random() < 0.5
    if max_jump == 1:
        jump = 1
    else:
        jump = int(rng.integers(1, max_jump + 1))

    nxt = state - jump if left else state + jump

    if nxt < 0:
        return (-1, -1.0, True)
    if nxt >= n_states:
        return (n_states, 1.0, True)
    return (int(nxt), 0.0, False)

# ── Step 003  generate_random_walk_episode ──
import numpy as np

def generate_random_walk_episode(n_states, rng, max_jump=1):
    state = n_states // 2
    states = []
    rewards = []
    while True:
        nxt, reward, done = random_walk_step(state, n_states, rng, max_jump)
        states.append(state)
        rewards.append(reward)
        if done:
            break

        state = nxt

    return (states, rewards)

# ── Step 004  n_step_return ──
import numpy as np

def n_step_return(rewards, states, t, n, values, gamma=1.0):
    T = len(rewards)
    G = 0.0
    discount = 1.0
    for k in range(n):
        idx = t + k
        if idx >= T:
            return float(G)

        G += discount * rewards[idx]
        discount *= gamma

    if t + n < T:
        G += discount * float(values[states[t + n]])

    return float(G)

# ── Step 005  n_step_td_episode ──
import numpy as np

def n_step_td_episode(values, states, rewards, n, alpha, gamma=1.0):
    v = np.array(values, dtype=float, copy=True)
    T = len(rewards)
    for t in range(T):
        G = n_step_return(rewards, states, t, n, v, gamma)
        s = states[t]
        v[s] += alpha * (G - v[s])

    return v

# ── Step 006  n_step_td_prediction ──
import numpy as np

def n_step_td_prediction(n_states, n, alpha, n_episodes, rng, gamma=1.0):
    true_values = random_walk_true_values(n_states)
    values = np.zeros(n_states, dtype=float)
    rms_errors = np.zeros(n_episodes, dtype=float)

    for ep in range(n_episodes):
        states, rewards = generate_random_walk_episode(n_states, rng)
        values = n_step_td_episode(values, states, rewards, n, alpha, gamma)
        rms_errors[ep] = np.sqrt(np.mean((values - true_values) ** 2))

    return values, rms_errors

# ── Step 007  n_step_parameter_sweep ──
import numpy as np

def n_step_parameter_sweep(n_states, ns, alphas, n_episodes, n_runs, seed):
    errors = np.zeros((len(ns), len(alphas)), dtype=float)
    for i, n in enumerate(ns):
        for j, alpha in enumerate(alphas):
            total = 0.0
            for run in range(n_runs):
                rng = np.random.default_rng(seed + run)
                _, rms = n_step_td_prediction(n_states, n, alpha, n_episodes, rng)
                total += float(np.mean(rms)) / n_runs

            errors[i, j] = total

    return errors

# ── Step 008  lambda_return ──
import numpy as np

def lambda_return(rewards, states, t, lam, values, gamma=1.0):
    T = len(rewards)
    n_max = T - t
    G = 0.0
    for n in range(1, n_max):
        G += (1.0 - lam) * (lam ** (n - 1)) * n_step_return(rewards, states, t, n, values, gamma)
        
    G += (lam ** (n_max - 1)) * n_step_return(rewards, states, t, n_max, values, gamma)
    return float(G)

# ── Step 009  offline_lambda_return_episode ──
import numpy as np

def offline_lambda_return_episode(values, states, rewards, lam, alpha, gamma=1.0):
    v = np.array(values, dtype=float, copy=True)
    T = len(rewards)
    targets = [0.0] * T

    G = 0.0
    for t in range(T - 1, -1, -1):
        next_v = v[states[t + 1]] if t + 1 < T else 0.0
        G = rewards[t] + gamma * ((1.0 - lam) * next_v + lam * G)
        targets[t] = G

    for t in range(T):
        s = states[t]
        v[s] += alpha * (targets[t] - v[s])

    return v

# ── Step 010  update_traces ──
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

# ── Step 011  td_lambda_episode ──
import numpy as np

def td_lambda_episode(values, states, rewards, lam, alpha, gamma, mode):
    v = np.array(values, dtype=float, copy=True)
    traces = np.zeros_like(v)
    T = len(rewards)

    for t in range(T):
        s = states[t]
        next_v = v[states[t + 1]] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - v[s]
        traces = update_traces(traces, s, lam, gamma, mode)
        v = v + alpha * delta * traces

    return v

# ── Step 012  lambda_prediction ──
import numpy as np

def lambda_prediction(n_states, lam, alpha, n_episodes, method, rng, gamma=1.0):
    true_values = random_walk_true_values(n_states)
    values = np.zeros(n_states, dtype=float)
    rms_errors = np.zeros(n_episodes, dtype=float)

    for ep in range(n_episodes):
        states, rewards = generate_random_walk_episode(n_states, rng)
        if method == "offline":
            values = offline_lambda_return_episode(values, states, rewards, lam, alpha, gamma)
        else:
            values = td_lambda_episode(values, states, rewards, lam, alpha, gamma, method)

        rms_errors[ep] = np.sqrt(np.mean((values - true_values) ** 2))

    return values, rms_errors

# ── Step 013  large_walk_true_values ──
import numpy as np

def large_walk_true_values(n_states, max_jump):
    P = np.zeros((n_states, n_states), dtype=float)
    r = np.zeros(n_states, dtype=float)

    for s in range(n_states):
        for j in range(1, max_jump + 1):
            p = 0.5 / max_jump
            left = s - j
            right = s + j

            if left < 0:
                r[s] += p * (-1.0)
            else:
                P[s, left] += p

            if right >= n_states:
                r[s] += p * (1.0)
            else:
                P[s, right] += p

    A = np.eye(n_states) - P
    v = np.linalg.solve(A, r)
    return v

# ── Step 014  state_aggregation_features ──
import numpy as np

def state_aggregation_features(state, n_states, n_groups):
    x = np.zeros(n_groups, dtype=np.float64)
    x[state * n_groups // n_states] = 1.0
    return x

# ── Step 015  gradient_mc_episode ──
import numpy as np

def gradient_mc_episode(weights, states, rewards, n_states, n_groups, alpha, gamma=1.0):
    w = np.array(weights, dtype=float, copy=True)
    T = len(rewards)

    returns = [0.0] * T
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    for t in range(T):
        x = state_aggregation_features(states[t], n_states, n_groups)
        w += alpha * (returns[t] - w @ x) * x

    return w

# ── Step 016  semi_gradient_td0_episode ──
import numpy as np

def semi_gradient_td0_episode(weights, states, rewards, n_states, n_groups, alpha, gamma=1.0):
    w = np.array(weights, dtype=float, copy=True)
    T = len(rewards)

    for t in range(T):
        x = state_aggregation_features(states[t], n_states, n_groups)
        if t + 1 < T:
            x_next = state_aggregation_features(states[t + 1], n_states, n_groups)
            next_value = w @ x_next
        else:
            next_value = 0.0
        delta = rewards[t] + gamma * next_value - w @ x
        w = w + alpha * delta * x

    return w

# ── Step 017  compare_value_approximations ──
import numpy as np

def compare_value_approximations(n_states, n_groups, max_jump, alpha_mc, alpha_td, n_episodes, seed):
    rng = np.random.default_rng(seed)
    true_values = large_walk_true_values(n_states, max_jump)

    w_mc = np.zeros(n_groups, dtype=float)
    w_td = np.zeros(n_groups, dtype=float)

    for _ in range(n_episodes):
        states, rewards = generate_random_walk_episode(n_states, rng, max_jump)
        w_mc = gradient_mc_episode(w_mc, states, rewards, n_states, n_groups, alpha_mc)
        w_td = semi_gradient_td0_episode(w_td, states, rewards, n_states, n_groups, alpha_td)

    groups = np.arange(n_states) * n_groups // n_states
    mc_values = w_mc[groups]
    td_values = w_td[groups]

    mc_rms = float(np.sqrt(np.mean((mc_values - true_values) ** 2)))
    td_rms = float(np.sqrt(np.mean((td_values - true_values) ** 2)))

    return {
        "mc_values": mc_values,
        "td_values": td_values,
        "mc_rms": mc_rms,
        "td_rms": td_rms,
    }

# ── Step 018  mountain_car_step ──
import math
import numpy as np

def mountain_car_step(position, velocity, action):
    velocity = velocity + 0.001 * (action - 1) - 0.0025 * math.cos(3.0 * position)
    velocity = max(-0.07, min(0.07, velocity))

    position = position + velocity
    position = max(-1.2, min(0.5, position))

    if position <= -1.2:
        velocity = 0.0

    done = bool(position >= 0.5)
    reward = -1.0

    return (float(position), float(velocity), reward, done)

# ── Step 019  tile_code ──
import numpy as np

def tile_code(position, velocity, n_tilings, tiles_per_dim):
    pos_scaled = (position + 1.2) / 1.7 * tiles_per_dim
    vel_scaled = (velocity + 0.07) / 0.14 * tiles_per_dim

    width = tiles_per_dim + 1
    out = np.empty(n_tilings, dtype=int)
    for i in range(n_tilings):
        offset = i / n_tilings
        px = min(int(np.floor(pos_scaled + offset)), tiles_per_dim)
        vx = min(int(np.floor(vel_scaled + offset)), tiles_per_dim)
        out[i] = i * width * width + px * width + vx
        
    return out

# ── Step 020  tile_epsilon_greedy ──
import numpy as np

def tile_epsilon_greedy(weights, active_tiles, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, weights.shape[0]))
    q = weights[:, active_tiles].sum(axis=1)
    return int(np.argmax(q))

# ── Step 021  sarsa_lambda_episode ──
import numpy as np

def sarsa_lambda_episode(weights, n_tilings, tiles_per_dim, lam, alpha, gamma, epsilon, rng, max_steps):
    w = np.array(weights, dtype=float, copy=True)
    traces = np.zeros_like(w)

    position = float(rng.uniform(-0.6, -0.4))
    velocity = 0.0

    tiles = tile_code(position, velocity, n_tilings, tiles_per_dim)
    action = tile_epsilon_greedy(w, tiles, epsilon, rng)

    for step in range(1, max_steps + 1):
        q_sa = w[action, tiles].sum()

        position, velocity, reward, done = mountain_car_step(position, velocity, action)

        delta = reward - q_sa

        traces *= gamma * lam
        traces[action, tiles] = 1.0

        if done:
            w += alpha * delta * traces
            return w, step

        new_tiles = tile_code(position, velocity, n_tilings, tiles_per_dim)
        new_action = tile_epsilon_greedy(w, new_tiles, epsilon, rng)
        delta += gamma * w[new_action, new_tiles].sum()

        w += alpha * delta * traces

        tiles = new_tiles
        action = new_action

    return w, max_steps

# ── Step 022  mountain_car_training ──
import numpy as np

def mountain_car_training(n_episodes, n_tilings, tiles_per_dim, lam, alpha, gamma, epsilon, seed, max_steps):
    rng = np.random.default_rng(seed)
    weights = np.zeros((3, n_tilings * (tiles_per_dim + 1) ** 2), dtype=float)
    steps_per_episode = np.zeros(n_episodes, dtype=int)

    for ep in range(n_episodes):
        weights, n = sarsa_lambda_episode(
            weights, n_tilings, tiles_per_dim, lam, alpha, gamma, epsilon, rng, max_steps
        )
        steps_per_episode[ep] = n

    return weights, steps_per_episode

# ── Scaffold (runner) ──
"""Sutton & Barto from scratch 3: n-step TD, eligibility traces and function approximation demo."""
import numpy as np


def main():
    # --- Part 1: n-step TD on the 19-state random walk (book figure 7.2 at small scale) ---
    n_states = 19
    true_values = random_walk_true_values(n_states)
    print("19-state true values (first 5):", np.round(true_values[:5], 2).tolist())

    rng = np.random.default_rng(0)
    states, rewards = generate_random_walk_episode(n_states, rng)
    print("Sample episode length:", len(states), "exit reward:", rewards[-1])
    print("n-step returns at t=0 for n=1,4,16:",
          [round(n_step_return(rewards, states, 0, n, true_values), 4) for n in (1, 4, 16)])

    ns = [1, 2, 4, 8, 16]
    alphas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    errors = n_step_parameter_sweep(n_states, ns, alphas, n_episodes=10, n_runs=20, seed=0)
    print("n-step sweep: mean RMS over 10 episodes x 20 runs (rows n, columns alpha)")
    for i, n in enumerate(ns):
        best = int(np.argmin(errors[i]))
        print(f"  n={n:>2}: {np.round(errors[i], 3).tolist()}  best alpha={alphas[best]} RMS={errors[i, best]:.4f}")
    best_n = ns[int(np.argmin(errors.min(axis=1)))]
    print("Best n overall (intermediate n wins, as in figure 7.2):", best_n)

    # --- Part 2: lambda-returns and eligibility traces (book figures 12.3 and 12.6) ---
    states, rewards = [2, 3, 2, 3, 4], [0.0, 0.0, 0.0, 0.0, 1.0]
    print("lambda-return at t=0 for lambda=0,0.5,1 on a toy episode:",
          [round(lambda_return(rewards, states, 0, lam, np.array([0.0, 0.0, 0.2, 0.4, 0.6])), 4) for lam in (0.0, 0.5, 1.0)])

    rng = np.random.default_rng(3)
    states, rewards = generate_random_walk_episode(n_states, rng)
    td0 = n_step_td_episode(np.zeros(n_states), states, rewards, 1, 0.3)
    tdl0 = td_lambda_episode(np.zeros(n_states), states, rewards, 0.0, 0.3, 1.0, 'accumulating')
    mc = n_step_td_episode(np.zeros(n_states), states, rewards, 10 ** 6, 0.3)
    off1 = offline_lambda_return_episode(np.zeros(n_states), states, rewards, 1.0, 0.3)
    print("TD(lambda=0) == TD(0):", bool(np.allclose(td0, tdl0)),
          "| offline lambda=1 == constant-alpha MC:", bool(np.allclose(mc, off1)))

    lams = [0.0, 0.4, 0.8, 0.9, 1.0]
    print("lambda study: mean RMS over 10 episodes x 20 runs, alpha=0.3 (columns lambda =", lams, ")")
    for method in ('offline', 'accumulating', 'replacing'):
        row = []
        for lam in lams:
            total = 0.0
            for run in range(20):
                _, rms = lambda_prediction(n_states, lam, 0.3, 10, method, np.random.default_rng(100 + run))
                total += rms.mean()
            mean_rms = float(total / 20)
            row.append(round(mean_rms, 3) if mean_rms < 10 else 'diverged')
        print(f"  {method:>12}: {row}")

    # --- Part 3: 1000-state random walk with state aggregation (book figures 9.1 / 9.2) ---
    big = large_walk_true_values(1000, 100)
    print("1000-state true values at 0, 250, 500, 750, 999:", np.round(big[[0, 250, 500, 750, 999]], 3).tolist())
    print("Feature of state 250 (10 groups):", state_aggregation_features(250, 1000, 10).astype(int).tolist())
    for n_episodes in (500, 3000):
        result = compare_value_approximations(1000, 10, 100, alpha_mc=0.0005, alpha_td=0.01,
                                              n_episodes=n_episodes, seed=0)
        print(f"After {n_episodes} episodes: gradient MC RMS={result['mc_rms']:.4f}  semi-gradient TD(0) RMS={result['td_rms']:.4f}")
    print("(TD(0) learns faster early; gradient MC ends closer to the true curve, as in figures 9.1/9.2)")
    print("Group centres   :", np.round(big[50::100], 2).tolist())
    print("MC staircase    :", np.round(result['mc_values'][50::100], 2).tolist())
    print("TD(0) staircase :", np.round(result['td_values'][50::100], 2).tolist())

    # --- Part 4: Mountain Car with tile coding and Sarsa(lambda) (book figure 10.1) ---
    position, velocity, reward, done = mountain_car_step(-0.5, 0.0, 2)
    print("One Mountain Car step from (-0.5, 0) with full throttle:", round(position, 4), round(velocity, 4), reward, done)
    print("Active tiles at (-0.5, 0):", tile_code(-0.5, 0.0, 8, 8).tolist())
    weights, steps = mountain_car_training(n_episodes=50, n_tilings=8, tiles_per_dim=8, lam=0.9,
                                           alpha=0.5 / 8, gamma=1.0, epsilon=0.0, seed=0, max_steps=5000)
    print("Steps per episode (first 10):", steps[:10].tolist())
    print("Steps per episode (last 10) :", steps[-10:].tolist())
    print("Mean steps, episodes 1-10 vs 41-50:", round(float(steps[:10].mean()), 1), "->", round(float(steps[-10:].mean()), 1))
    start_tiles = tile_code(-0.5, 0.0, 8, 8)
    print("Learned q(start, a) for a=0,1,2:", np.round(weights[:, start_tiles].sum(axis=1), 2).tolist())


if __name__ == "__main__":
    main()
