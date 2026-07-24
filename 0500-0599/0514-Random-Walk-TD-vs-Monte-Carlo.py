def random_walk_td_mc(episodes: list, n_states: int, gamma: float = 1.0, alpha: float = 0.1) -> tuple:
    """
    Run TD(0) and first-visit Monte Carlo prediction on random walk episodes.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of non-terminal states
        gamma: Discount factor
        alpha: Learning rate for TD(0)
    
    Returns:
        Tuple of (td_values, mc_values) as lists of floats
    """
    td_values = [0.0] * n_states

    for episode in episodes:
        for state, reward, next_state, done in episode:
            if done:
                target = reward
            else:
                target = reward + gamma * td_values[next_state]

            td_values[state] += alpha * (target - td_values[state])

    mc_values = [0.0] * n_states
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states

    for episode in episodes:
        visited = set()
        G = 0.0
        for state, reward, next_state, done in reversed(episode):
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
    
    for s in range(n_states):
        if returns_count[s] > 0:
            mc_values[s] = returns_sum[s] / returns_count[s]
    
    return td_values, mc_values
