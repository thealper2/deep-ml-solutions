import numpy as np

def check_anti_diagonal_win(board, player):
    return np.all(np.diag(np.fliplr(board)) == player)
