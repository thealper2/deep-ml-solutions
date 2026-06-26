import numpy as np

def encode_board_one_hot_length_eighteen(board, current_player):
    """Encode a 3x3 board as a length-18 two-channel one-hot vector."""
    flipped = flip_board_perspective(board, current_player)
    own = (flipped == 1).astype(np.float32).flatten()
    opp = (flipped == -1).astype(np.float32).flatten()
    return np.concatenate([own, opp])
