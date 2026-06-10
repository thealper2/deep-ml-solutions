def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        total_reward += reward
        if done:
            break

    return total_reward
