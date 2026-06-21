import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return torch.where(mask, scores, torch.tensor(float('-inf')))
