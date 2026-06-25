def episode_apply_q_update(q_table, state_key, action, reward, next_board, done, alpha, gamma):
    """Compute the TD target (terminal or nonterminal) and apply the Q-learning update."""
    if done:
        target = q_learning_terminal_target(reward)
    else:
        next_state_key = canonical_board_key(next_board)
        next_legal_actions = get_legal_moves(next_board)
        target = q_learning_nonterminal_target(reward, gamma, q_table, next_state_key, next_legal_actions)
    
    old_q = get_q_value(q_table, state_key, action)
    new_q = old_q + alpha * (target - old_q)
    q_table[(state_key, action)] = new_q
    return new_q
