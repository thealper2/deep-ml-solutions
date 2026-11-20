import torch
import torch.nn as nn


class ToolLM(nn.Module):
    """A small causal transformer: token ids (B, T) in, next-token logits (B, T, vocab_size) out."""

    def __init__(self, vocab_size, block_size=40, d_model=48, n_head=4, n_layer=1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_head, 4 * d_model,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=causal)
        x = self.ln_f(x)
        return self.head(x)