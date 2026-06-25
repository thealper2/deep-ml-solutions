def minimax_recursive(board, player):
    """Return the minimax value of `board` with `player` to move."""
    cache = {}
    key = (board.tobytes(), player)
    if key in cache:
        return cache[key]
    
    status = get_game_status(board)
    if status != 'ongoing':
        value = minimax_terminal_score(status)
        cache[key] = value
        return value
    
    moves = get_legal_moves(board)
    if player == 1:
        best = -float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_recursive(new_board, switch_player(player))
            best = max(best, value)
        cache[key] = best
        return best
    else:
        best = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_recursive(new_board, switch_player(player))
            best = min(best, value)
        cache[key] = best
        return best
