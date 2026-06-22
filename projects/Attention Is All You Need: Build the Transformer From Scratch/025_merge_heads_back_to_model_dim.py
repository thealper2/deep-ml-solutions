import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    B, num_heads, L, d_k = multi_head_tensor.shape
    return multi_head_tensor.transpose(1, 2).contiguous().view(B, L, num_heads * d_k)
