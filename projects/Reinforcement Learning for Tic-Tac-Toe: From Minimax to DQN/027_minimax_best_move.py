def minimax_best_move(board, player):
    """Return the optimal (row, col) move for `player` via minimax."""
    _, best_move = minimax_max_min_step(board, player)
    return best_move
