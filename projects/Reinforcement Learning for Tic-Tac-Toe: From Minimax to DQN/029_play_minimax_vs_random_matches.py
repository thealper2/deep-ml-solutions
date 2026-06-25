def play_minimax_vs_random_matches(n_games, minimax_plays_x, rng):
    outcomes = []

    for _ in range(n_games):
        game = TicTacToeGame()

        while not game.is_terminal():
            board = game.board
            player = game.current_player

            if (minimax_plays_x and player == 1) or \
               (not minimax_plays_x and player == -1):
                _, move = minimax_max_min_step(board, player)
            else:
                move = random_move_agent(board, player, rng)

            row, col = move
            game.step(row, col)

        outcomes.append(game.status)

    return compute_outcome_rates(outcomes)
