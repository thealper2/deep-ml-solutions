import numpy as np

def is_draw(board):
    """Return True iff the board is full and neither player has won."""
    if len(get_legal_moves(board)) > 0:
        return False

    if is_winner(board, 1) or is_winner(board, -1):
        return False

    return True
