import torch

def encode_sequence(start: tuple, goal: tuple, moves: list, G: int, T: int) -> tuple:
    start_token = 4 + start[0] * G + start[1]
    goal_token = 4 + goal[0] * G + goal[1]
    EOS = 4 + G * G
    
    max_moves = T - 3
    if len(moves) > max_moves:
        moves = moves[:max_moves]
    
    tokens_list = [start_token, goal_token] + moves + [EOS]
    real_len = len(tokens_list)
    
    while len(tokens_list) < T:
        tokens_list.append(EOS)
    
    tokens = torch.tensor(tokens_list, dtype=torch.long)
    
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:real_len] = True
    
    return tokens, mask