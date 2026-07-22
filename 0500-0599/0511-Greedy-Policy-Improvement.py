def greedy_policy_improvement(V: dict, transitions: dict, gamma: float) -> tuple:
    """
    Perform greedy policy improvement given a value function and MDP model.
    
    Args:
        V: dict mapping state -> float (state-value function)
        transitions: dict mapping (state, action) -> list of (probability, next_state, reward)
        gamma: float, discount factor
    
    Returns:
        tuple: (policy, Q)
            policy: dict mapping state -> best action
            Q: dict mapping (state, action) -> float
    """
    policy = {}
    Q = {}

    for (state, action), outcomes in transitions.items():
        q_value = 0.0
        for prob, next_state, reward in outcomes:
            q_value += prob * (reward + gamma * V[next_state])

        Q[(state, action)] = q_value

    states = set(state for state, _ in Q.keys())
    for state in states:
        actions = [action for (s, action) in Q.keys() if s == state]
        best_action = max(actions, key=lambda a: (Q[(state, a)], -hash(a)))
        best_action = sorted(actions, key=lambda a: (-Q[(state, a)], a))[0]
        policy[state] = best_action

    return policy, Q
