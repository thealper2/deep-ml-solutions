def gambler_value_iteration(goal, head_prob, theta, gamma=1.0):
    """Solve the gambler's problem with value iteration.

    Parameters
    ----------
    goal : int
        Capital target (terminal winning state).
    head_prob : float
        Probability the coin lands heads.
    theta : float
        Stop when the largest value change in a sweep is below this.
    gamma : float, optional
        Discount factor (default 1.0).

    Returns
    -------
    state_values : np.ndarray, shape (goal+1,)
        Optimal values; state_values[0] and state_values[goal] are 0.
    """
    n_states = goal + 1
    V = np.zeros(n_states)
    V[goal] = 0.0
    
    while True:
        delta = 0.0
        V_new = V.copy()
        
        for s in range(1, goal):
            max_stake = min(s, goal - s)
            if max_stake <= 0:
                V_new[s] = 0.0
                continue
            
            best_value = -float('inf')
            for stake in range(1, max_stake + 1):
                win_state = s + stake
                win_reward = 1.0 if win_state == goal else 0.0
                win_value = head_prob * (win_reward + gamma * V[win_state])
                
                lose_state = s - stake
                lose_value = (1.0 - head_prob) * (0.0 + gamma * V[lose_state])
                
                q_value = win_value + lose_value
                best_value = max(best_value, q_value)
            
            V_new[s] = best_value
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    return V
