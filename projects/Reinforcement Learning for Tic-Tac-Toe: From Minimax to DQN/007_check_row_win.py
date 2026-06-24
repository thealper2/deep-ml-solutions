import numpy as np

def check_row_win(board, player):
    """Return True if `player` has three-in-a-row across any row of `board`."""
    for row in board:
        if np.all(row == player):
            return True

    return False
