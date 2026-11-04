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