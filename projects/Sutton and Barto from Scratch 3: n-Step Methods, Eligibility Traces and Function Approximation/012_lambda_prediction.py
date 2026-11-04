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