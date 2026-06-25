def epsilon_greedy_explore_move(legal_actions, rng):
    """Sample a uniformly random legal action from legal_actions using rng."""
    idx = rng.integers(0, len(legal_actions))
    return legal_actions[idx]
