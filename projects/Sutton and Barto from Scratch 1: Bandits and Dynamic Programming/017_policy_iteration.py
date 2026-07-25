def policy_iteration(mdp, gamma, theta):
    n_states = mdp['n_states']
    policy = np.zeros(n_states, dtype=int)

    while True:
        V = iterative_policy_evaluation(policy, mdp, gamma, theta)
        new_policy = greedy_policy_improvement(V, mdp, gamma)
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy
