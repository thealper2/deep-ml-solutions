import numpy as np

def state_aggregation_mc(
    episodes: list,
    n_states: int,
    group_assignments: list,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using state aggregation with Monte Carlo returns.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
        n_states: Total number of states (states are integers 0 to n_states-1).
        group_assignments: List of length n_states where group_assignments[s] is the
                          group index for state s.
        gamma: Discount factor.
        
    Returns:
        V: Estimated value for each state as numpy array of shape (n_states,).
    """
    n_groups = max(group_assignments) + 1
    group_returns = [[] for _ in range(n_groups)]

    for episode in episodes:
        T = len(episode)

        for start_idx in range(T):
            G = 0.0
            for t in range(start_idx, T):
                reward = episode[t][1]
                G += (gamma ** (t - start_idx)) * reward

            state = episode[start_idx][0]
            group = group_assignments[state]

            group_returns[group].append(G)

    group_values = [0.0] * n_groups
    for g in range(n_groups):
        if group_returns[g]:
            group_values[g] = sum(group_returns[g]) / len(group_returns[g])

    state_values = [group_values[group_assignments[s]] for s in range(n_states)]
    return state_values