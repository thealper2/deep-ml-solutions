import numpy as np

def encode_board_flat_length_nine(board, current_player):
    """Encode a 3x3 board as a length-9 float32 vector from current_player's view."""
    flipped = flip_board_perspective(board, current_player)
    return flipped.flatten().astype(np.float32)
