import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=64, heads=4, layers=2, max_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        for layer in self.layers:
            h = layer(h, src_mask=mask)

        h = self.norm(h)
        return self.out(h)