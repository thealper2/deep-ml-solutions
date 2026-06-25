import numpy as np

def get_current_player(board):
    """Return 1 if X is to move, -1 if O is to move."""
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)

    if x_count == o_count:
        return 1
    else:
        return -1
