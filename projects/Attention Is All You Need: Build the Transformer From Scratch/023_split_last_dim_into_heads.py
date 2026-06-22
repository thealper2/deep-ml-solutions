import torch

def split_last_dim_into_heads(tensor, num_heads):
    B, L, d_model = tensor.shape
    d_k = d_model // num_heads
    return tensor.view(B, L, num_heads, d_k)
