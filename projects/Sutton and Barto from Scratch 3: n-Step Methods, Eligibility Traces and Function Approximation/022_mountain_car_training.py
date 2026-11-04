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