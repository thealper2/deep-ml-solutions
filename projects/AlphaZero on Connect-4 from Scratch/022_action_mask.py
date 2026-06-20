import numpy as np

def action_mask(board):
    mask = np.zeros(7, dtype=bool)
    valid = valid_moves(board)
    for col in valid:
        mask[col] = True
        
    return mask
