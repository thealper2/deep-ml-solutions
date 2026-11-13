import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        """key, query and value projections, a registered tril buffer, and dropout on the attention weights."""
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, T, n_embd) -> (B, T, head_size): scaled, causally masked attention."""
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ self.value(x)