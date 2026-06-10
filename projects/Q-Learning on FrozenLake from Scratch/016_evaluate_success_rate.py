def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):
    successes = 0
    for i in range(num_episodes):
        episode_seed = seed + i
        if run_greedy_episode(env, policy, seed=episode_seed, max_steps=max_steps):
            successes += 1

    return successes / num_episodes
