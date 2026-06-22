import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.transpose(1, 2).contiguous()
