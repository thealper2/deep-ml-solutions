def value_iteration(mdp, gamma, theta):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']
    
    V = np.zeros(n_states)
    
    while True:
        delta = 0.0
        V_new = V.copy()
        
        for s in range(n_states):
            max_value = -float('inf')
            for a in range(n_actions):
                q_value = sum(prob * (reward + gamma * V[next_state])
                              for prob, next_state, reward in P[s][a])
                max_value = max(max_value, q_value)
            
            V_new[s] = max_value
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    policy = greedy_policy_improvement(V, mdp, gamma)
    return V, policy
