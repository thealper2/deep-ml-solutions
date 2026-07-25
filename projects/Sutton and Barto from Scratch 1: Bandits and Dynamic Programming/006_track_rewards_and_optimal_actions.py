def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    k = len(true_values)
    optimal_arm = int(np.argmax(true_values))

    q_values = np.zeros(k)
    action_counts = np.zeros(k, dtype=int)

    rewards = np.zeros(n_steps)
    optimal_flags = np.zeros(n_steps)

    for step in range(n_steps):
        if rng.random() < epsilon:
            action = rng.integers(0, k)
        else:
            action = int(np.argmax(q_values))

        reward = rng.normal(loc=true_values[action], scale=1.0)
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
        rewards[step] = reward
        optimal_flags[step]= 1.0 if action == optimal_arm else 0.0

    return rewards, optimal_flags
