def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    return next_state, float(reward), done
