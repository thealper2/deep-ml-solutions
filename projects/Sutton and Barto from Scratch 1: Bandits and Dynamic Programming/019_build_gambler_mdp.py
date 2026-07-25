def build_gambler_mdp(goal, head_prob):
    """Build the gambler's-problem MDP as a dynamics dictionary.

    Parameters
    ----------
    goal : int
        Capital target (terminal winning state).
    head_prob : float
        Probability that the coin lands heads.

    Returns
    -------
    mdp : dict
        Keys 'n_states', 'n_actions', and 'P' (dynamics table).
    """
    n_states = goal + 1
    n_actions = goal

    P = []

    for s in range(n_states):
        if s == 0 or s == goal:
            P.append([[(1.0, s, 0.0)]])
        else:
            legal_actions = []
            max_stake = min(s, goal - s)
            for stake in range(1, max_stake + 1):
                heads_reward = 1.0 if s + stake == goal else 0.0
                action_transitions = [
                    (head_prob, s + stake, heads_reward),
                    (1.0 - head_prob, s - stake, 0.0)
                ]
                legal_actions.append(action_transitions)

            P.append(legal_actions)

    return {
        'n_states': n_states,
        'n_actions': n_actions,
        'P': P,
    }
