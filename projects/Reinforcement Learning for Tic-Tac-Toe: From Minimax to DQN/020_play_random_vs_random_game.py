def play_random_vs_random_game(rng):
    """Simulate one full random-vs-random game and return the final status."""
    game = TicTacToeGame()

    while not game.is_terminal():
        moves = game.legal_moves()
        row, col = random_move_agent(game.board, moves, rng)
        game.step(row, col)

    return game.status
