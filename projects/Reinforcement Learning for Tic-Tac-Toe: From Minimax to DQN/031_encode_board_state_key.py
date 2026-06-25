import numpy as np

def encode_board_state_key(board):
    """Encode a 3x3 board as a length-9 string over {'0','1','2'} in row-major order."""
    d = {0: '0', 1: '1', -1: '2'}
    result = ''
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            result += d[board[row][col]]
                
    return result
