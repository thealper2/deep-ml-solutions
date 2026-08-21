import torch
import torch.nn.functional as F
import math

def causal_self_attention(x, w_q, w_k, w_v, w_o):
    """Compute single-head causal scaled-dot-product attention."""
    B, T, D = x.shape
    Q = x @ w_q
    K = x @ w_k
    V = x @ w_v
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    out = attn @ V
    out = out @ w_o
    return out