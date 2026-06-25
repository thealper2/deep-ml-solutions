import numpy as np

def minimax_max_min_step(board, player):
    """Return (best_score, best_move) after expanding one minimax level."""
    moves = get_legal_moves(board)
    best_score = None
    best_move = None

    if player == 1:
        best_score = float('-inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            score = minimax_recursive(new_board, switch_player(player))
            if score > best_score:
                best_score = score
                best_move = (row, col)

    else:
        best_score = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            score = minimax_recursive(new_board, switch_player(player))
            if score < best_score:
                best_score = score
                best_move = (row, col)

    return best_score, best_move
