def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    for _ in range(max_steps):
        action = int(policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            return reward > 0

        state = next_state

    return False
