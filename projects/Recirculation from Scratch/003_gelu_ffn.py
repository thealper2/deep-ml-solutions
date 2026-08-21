import torch
import torch.nn.functional as F

def gelu_ffn(x, w_ff1, w_ff2):
    """Apply a two-layer position-wise GELU feed-forward that expands to 4D then projects back to D."""
    h = x @ w_ff1
    h = F.gelu(h)
    out = h @ w_ff2
    return out