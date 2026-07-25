def greedy_policy_improvement(state_values, mdp, gamma):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']
    
    policy = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        best_action = 0
        best_value = -float('inf')
        
        for a in range(n_actions):
            q_value = sum(prob * (reward + gamma * state_values[next_state]) for prob, next_state, reward in P[s][a])
            
            if q_value > best_value:
                best_value = q_value
                best_action = a
        
        policy[s] = best_action
    
    return policy
