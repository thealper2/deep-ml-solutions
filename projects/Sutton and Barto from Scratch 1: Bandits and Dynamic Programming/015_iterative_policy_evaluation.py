def iterative_policy_evaluation(policy, mdp, gamma, theta):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']

    V = np.zeros(n_states)

    while True:
        delta = 0.0
        V_new = V.copy()

        for s in range(n_states):
            if policy.ndim == 1:
                a = policy[s]
                new_v = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in P[s][a])
            else:
                new_v = 0.0
                for a in range(n_actions):
                    prob_action = policy[s, a]
                    if prob_action > 0:
                        v_a = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in P[s][a])
                        new_v += prob_action * v_a

            V_new[s] = new_v
            delta = max(delta, abs(V_new[s] - V[s]))
    
        V = V_new
        if delta < theta:
            break

    return V
