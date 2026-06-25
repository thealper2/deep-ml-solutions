import numpy as np

def episode_reset_game():
    """Return a fresh empty board and the starting player (+1 for X)."""
    board = create_empty_board()
    player = 1
    return board, player
