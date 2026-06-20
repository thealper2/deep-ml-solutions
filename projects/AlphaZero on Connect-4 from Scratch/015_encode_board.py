def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    opponent = other_player(current_player)
    tensor = np.zeros((2, 6, 7), dtype=np.float32)
    tensor[0] = (board == current_player).astype(np.float32)
    tensor[1] = (board == opponent).astype(np.float32)
    return tensor
