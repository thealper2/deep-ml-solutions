import numpy as np

def is_winner(board, player):
    """Return True if `player` has three-in-a-row on `board`."""
    if check_row_win(board, player):
        return True

    if check_column_win(board, player):
        return True

    if check_main_diagonal_win(board, player):
        return True

    if check_anti_diagonal_win(board, player):
        return True

    return False
