import numpy as np

def print_board(board):
    """Print the 3x3 board using X, O, and . characters."""
    for row in board:
        line = []
        for cell in row:
            if cell == 1:
                line.append('X')
            elif cell == -1:
                line.append('O')
            else:
                line.append('.')

        print(' '.join(line))
