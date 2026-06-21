import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    return torch.matmul(query, key.transpose(-2, -1))
