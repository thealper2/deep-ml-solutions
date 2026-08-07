import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                dropout=0.0
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_emb(idx)

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(positions)

        x = tok_emb + pos_emb

        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))

        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits
