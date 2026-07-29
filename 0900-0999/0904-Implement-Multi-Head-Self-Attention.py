import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, d_model = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(B, T, d_model)

        out = self.out_proj(out)
        return out
