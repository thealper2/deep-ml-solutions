def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    k = len(true_values)
    q_values = np.zeros(k)
    action_counts = np.zeros(k, dtype=int)

    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        action = epsilon_greedy_action(q_values, epsilon, rng)
        reward = rng.normal(loc=true_values[action], scale=1.0)
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
        rewards[step] = reward
        actions[step] = action

    return rewards, actions
