def canonical_board_key(board):
    rotate_90 = lambda x: np.rot90(x)
    reflect = lambda x: np.fliplr(x)

    candidates = []
    current = board.copy()
    for _ in range(4):
        candidates.append(current)
        current = rotate_90(current)

    reflected = reflect(board)
    current = reflected.copy()
    for _ in range(4):
        candidates.append(current)
        current = rotate_90(current)

    keys = [encode_board_state_key(b) for b in candidates]
    return min(keys)
