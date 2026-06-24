import numpy as np

def check_main_diagonal_win(board, player):
    """Return True if `player` occupies all three main-diagonal cells."""
    return np.all(np.diag(board) == player)
