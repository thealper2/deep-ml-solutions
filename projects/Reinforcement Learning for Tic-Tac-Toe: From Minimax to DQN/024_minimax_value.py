def minimax_value(board, player):
    """Return the minimax value of `board` with `player` to move."""
    status = get_game_status(board)
    if status != 'ongoing':
        return minimax_terminal_score(status)

    moves = get_legal_moves(board)
    if player == 1:
        best = float('-inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_value(new_board, switch_player(player))
            best = max(best, value)
        return best
    else:
        best = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_value(new_board, switch_player(player))
            best = min(best, value)
        return best
