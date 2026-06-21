def evaluate_against_random(net, num_matches, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    greedy_agent = lambda state, to_play: greedy_agent_action(net, state, to_play)
    random_agent = lambda state, to_play: random_policy_action(state, to_play, rng)

    return match_win_rate(greedy_agent, random_agent, num_matches, alternate_starts=True)
