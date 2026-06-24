import numpy as np

def get_legal_moves(board):
    """Return a list of (row, col) tuples for all empty cells on the board."""
    row, col = board.shape
    result = []

    for r in range(row):
        for c in range(col):
            if board[r][c] == 0:
                result.append((r, c))

    return result
