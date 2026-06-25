def play_minimax_vs_minimax_matches(n_games):
    """Play n_games minimax-vs-minimax games and report outcome rates plus an all_draws flag."""
    outcomes = []
    
    for _ in range(n_games):
        game = TicTacToeGame()
        
        while not game.is_terminal():
            board = game.board
            player = game.current_player
            
            _, move = minimax_alpha_beta(board, player, -float('inf'), float('inf'))
            row, col = move
            game.step(row, col)
        
        outcomes.append(game.status)
    
    rates = compute_outcome_rates(outcomes)
    rates['all_draws'] = rates['draw_rate'] == 1.0 if n_games > 0 else True
    
    return rates
