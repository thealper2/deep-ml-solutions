import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return causal_mask.unsqueeze(0).unsqueeze(0)
