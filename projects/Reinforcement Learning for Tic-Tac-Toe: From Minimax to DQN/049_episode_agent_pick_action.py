def episode_agent_pick_action(q_table, board, current_player, epsilon, rng):
    state_key = canonical_board_key(board)
    moves = get_legal_moves(board)
    legal_actions = [row * 3 + col for row, col in moves]
    action = epsilon_greedy_select_action(q_table, state_key, legal_actions, epsilon, rng)
    return state_key, action
