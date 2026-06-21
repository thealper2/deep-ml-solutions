def encode_batch_states(boards, to_plays):
    encoded_boards = []
    for board, to_play in zip(boards, to_plays):
        enc = encode_board(board, to_play)
        encoded_boards.append(torch.tensor(enc, dtype=torch.float32))

    return torch.stack(encoded_boards)
