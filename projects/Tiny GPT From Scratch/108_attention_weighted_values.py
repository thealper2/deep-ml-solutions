import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    return attn @ v
