import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
