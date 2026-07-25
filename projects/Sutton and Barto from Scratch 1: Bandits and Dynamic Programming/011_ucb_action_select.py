def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    k = len(q_values)
    best_action = None
    best_score = -float('inf')
    
    ln_t = np.log(timestep)
    
    for action in range(k):
        if action_counts[action] == 0:
            return action
        
        score = q_values[action] + c * np.sqrt(ln_t / action_counts[action])
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action
