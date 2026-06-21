import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    mask = (token_ids != pad_id).unsqueeze(1).unsqueeze(2)
    return mask
