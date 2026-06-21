import torch

def softmax_attention_weights(masked_scores):
    all_masked = (masked_scores == float('-inf')).all(dim=-1, keepdim=True)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.where(all_masked, torch.zeros_like(weights), weights)
    return weights
