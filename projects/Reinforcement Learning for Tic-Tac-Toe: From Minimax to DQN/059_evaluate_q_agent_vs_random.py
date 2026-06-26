def evaluate_q_agent_vs_random(q_table, num_games, rng):
    """Play num_games between the greedy Q-agent and a random opponent.

    Returns a dict with keys 'wins', 'losses', 'draws' (ints) and
    'win_rate', 'loss_rate', 'draw_rate' (floats), all from the agent's
    perspective. The agent alternates between playing X and O across games.
    """
    wins = 0
    losses = 0
    draws = 0
    
    for game_idx in range(num_games):
        game = TicTacToeGame()
        
        agent_player = 1 if game_idx % 2 == 0 else -1
        agent_side = 'X' if agent_player == 1 else 'O'
        
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
                legal_moves = get_legal_moves(board)
                action = random_move_agent(board, current_player, rng)
                row, col = action
                action_flat = row * 3 + col
                out = episode_apply_action(board, action_flat, current_player, agent_player)
            
            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']
        
        status = game.status
        if status == 'X_win':
            if agent_player == 1:
                wins += 1
            else:
                losses += 1
        elif status == 'O_win':
            if agent_player == -1:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1
    
    total = num_games
    if total == 0:
        return {'wins': 0, 'losses': 0, 'draws': 0, 'win_rate': 0.0, 'loss_rate': 0.0, 'draw_rate': 0.0}
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / total,
        'loss_rate': losses / total,
        'draw_rate': draws / total
    }
