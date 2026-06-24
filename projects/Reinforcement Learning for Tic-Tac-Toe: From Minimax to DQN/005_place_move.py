import numpy as np

def place_move(board, row, col, player):
    """Place player's mark at (row, col) and return the new board."""
    if not is_cell_empty(board, row, col):
        raise ValueError('Cell is already occupied')

    new_board = board.copy()
    new_board[row][col] = player
    return new_board
