import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        """Multi-head attention with n_head heads of n_embd // n_head, a feed-forward layer, and two layer norms."""
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        """Pre-norm residual: x + sa(ln1(x)), then x + ffwd(ln2(x))."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x