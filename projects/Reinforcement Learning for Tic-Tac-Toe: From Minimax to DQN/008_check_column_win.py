import numpy as np

def check_column_win(board, player):
    """Return True if `player` has three-in-a-row in any column of `board`."""
    for col in range(board.shape[1]):
        if np.all(board[:, col] == player):
            return True

    return False
