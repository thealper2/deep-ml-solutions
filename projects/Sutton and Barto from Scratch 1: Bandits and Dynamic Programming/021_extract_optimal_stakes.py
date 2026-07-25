def extract_optimal_stakes(state_values, goal, head_prob, gamma=1.0):
    """Extract the optimal stake for every capital level from V.

    Parameters
    ----------
    state_values : np.ndarray, shape (goal + 1,)
        Converged state values for capitals 0 .. goal.
    goal : int
        Capital target.
    head_prob : float
        Probability the coin lands heads.
    gamma : float, optional
        Discount factor (default 1.0).

    Returns
    -------
    stakes : np.ndarray, shape (goal + 1,), dtype int
        stakes[s] is an optimal stake for capital s (0 at terminals).
        Ties are broken by choosing the smallest stake.
    """
    stakes = np.zeros(goal + 1, dtype=int)

    for s in range(1, goal):
        max_stake = min(s, goal - s)
        best_value = float('-inf')
        best_stake = 1

        for stake in range(1, max_stake + 1):
            win_state = s + stake
            win_reward = 1.0 if win_state == goal else 0.0
            win_value = head_prob * (win_reward + gamma * state_values[win_state])

            lose_state = s - stake
            lose_value = (1.0 - head_prob) * (0.0 + gamma * state_values[lose_state])

            q_value = win_value + lose_value

            if q_value > best_value:
                best_value = q_value
                best_stake = stake

        stakes[s] = best_stake
    
    return stakes
