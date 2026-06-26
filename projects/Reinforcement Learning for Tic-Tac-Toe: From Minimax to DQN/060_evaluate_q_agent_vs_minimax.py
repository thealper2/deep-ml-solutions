def evaluate_q_agent_vs_minimax(q_table, num_games, rng):
    outcomes = []

    for game_idx in range(num_games):
        game = TicTacToeGame()

        agent_player = 1 if game_idx % 2 == 0 else -1

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                state_key = canonical_board_key(board)
                legal_moves = get_legal_moves(board)
                legal_actions = [row * 3 + col for row, col in legal_moves]
                action = greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng)
                out = episode_apply_action(board, action, current_player, agent_player)

            else:
                score, move = minimax_alpha_beta(board, current_player, float('inf'), float('-inf'))
                row, col = move
                action_flat = row * 3 + col
                out = episode_apply_action(board, action_flat, current_player, agent_player)

            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']

        status = game.status
        if status == 'X_win':
            outcome = 'win' if agent_player == 1 else 'loss'
        elif status == 'Q_win':
            outcome = 'win' if agent_player == 0 else 'loss'
        else:
            outcome = 'draw'

        outcomes.append(outcome)

    return compute_outcome_rates(outcomes)
