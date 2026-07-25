def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    total_rewards = np.zeros(n_steps)
    total_optimal = np.zeros(n_steps)

    for run_idx in range(n_runs):
        true_values = create_bandit_testbed(k, seed + run_idx)
        rng = np.random.default_rng(seed + run_idx)
        rewards, optimal_flags = track_rewards_and_optimal_actions(
            true_values, n_steps, epsilon, rng
        )
        total_rewards += rewards
        total_optimal += optimal_flags

    mean_rewards = total_rewards / n_runs
    mean_optimal = total_optimal / n_runs
    return mean_rewards, mean_optimal
