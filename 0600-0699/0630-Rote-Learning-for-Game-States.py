def rote_learning(episodes: list, gamma: float) -> dict:
    """
    Build a rote-learning value table from game episodes.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
        gamma: Discount factor.
    
    Returns:
        Dictionary mapping each state to its average discounted return,
        rounded to 4 decimal places.
    """
    returns_dict = {}
    for episode in episodes:
        T = len(episode)
        G = 0.0
        for t in range(T - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            if state not in returns_dict:
                returns_dict[state] = []

            returns_dict[state].append(G)

    result = {}
    for state, returns in returns_dict.items():
        avg_return = sum(returns) / len(returns)
        result[state] = round(avg_return, 4)

    return result