import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        """num_heads Heads in an nn.ModuleList, a projection back to n_embd, and dropout."""
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Concatenate the heads' outputs, project, dropout."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))