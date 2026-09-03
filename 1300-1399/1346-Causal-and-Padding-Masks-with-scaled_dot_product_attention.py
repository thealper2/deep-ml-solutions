import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]

    attn_mask = None

    if causal:
        causal_mask = torch.tril(torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device))
        attn_mask = causal_mask if attn_mask is None else attn_mask & causal_mask

    if key_padding_mask is not None:
        pad_mask = key_padding_mask.view(batch, 1, 1, seq_k)
        attn_mask = pad_mask if attn_mask is None else attn_mask & pad_mask

    if attn_mask is None:
        return F.scaled_dot_product_attention(q, k, v)
    else:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
