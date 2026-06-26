def dqn_select_action(online_params, state, legal_mask, epsilon, rng):
    """Epsilon-greedy action index over the legal moves."""
    if rng.random() < epsilon:
        legal_indices = np.where(legal_mask)[0]
        idx = rng.integers(0, len(legal_indices))
        return int(legal_indices[idx])
    else:
        q_values, _ = mlp_forward_pass(online_params, state.reshape(1, -1))
        q_values = q_values.flatten()
        masked_q = mask_illegal_actions_neg_inf(q_values, legal_mask)
        return int(argmax_action_from_q_values(masked_q))
