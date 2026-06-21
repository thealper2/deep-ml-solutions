def mcts_choose_action(state, to_play, net, num_simulations, c_puct, temperature=1.0):
    root = run_mcts(state, to_play, net, num_simulations, c_puct)
    policy = visit_count_policy(root, temperature)
    
    if not np.isclose(policy.sum(), 1.0):
        policy = policy / policy.sum()
    
    policy_tensor = torch.tensor(policy, dtype=torch.float32)
    dist = torch.distributions.Categorical(policy_tensor)
    action = dist.sample().item()
    
    return action, policy
