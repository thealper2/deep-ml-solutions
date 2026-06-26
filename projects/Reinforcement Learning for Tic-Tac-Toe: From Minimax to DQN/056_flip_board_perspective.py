import numpy as np

def flip_board_perspective(board, current_player):
    """Return a board view where current_player's marks are +1."""
    return board.copy() if current_player == 1 else -board
