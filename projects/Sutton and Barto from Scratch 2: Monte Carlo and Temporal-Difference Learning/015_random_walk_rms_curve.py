import numpy as np

def random_walk_rms_curve(method, alpha, n_episodes, n_runs, rng):
    true_values = np.arange(1, 6, dtype=float) / 6.0
    rms_errors = np.zeros(n_episodes, dtype=float)

    for _ in range(n_runs):
        values = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=float)

        for episode_idx in range(n_episodes):
            transitions = random_walk_episode(rng)

            if method == 'td':
                values = td0_update(values, transitions, alpha, gamma=1.0)
            elif method == 'mc':
                values = constant_alpha_mc_update(values, transitions, alpha, gamma=1.0)
            else:
                raise ValueError("method must be 'td' or 'mc'")

            error = np.sqrt(np.mean((values[1:6] - true_values) ** 2))
            rms_errors[episode_idx] += error

    return rms_errors / n_runs