def board_to_torch_tensor(board, current_player):
    encoded = encode_board(board, current_player)
    return torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
