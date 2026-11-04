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