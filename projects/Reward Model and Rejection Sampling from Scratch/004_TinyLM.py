import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_head=2, n_layer=2, max_len=16):
        """Token and position embeddings, n_layer causal pre-norm blocks, a final LayerNorm, and self.lm_head."""
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.max_len = max_len
        self.head_size = d_model // n_head

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "qkv": nn.Linear(d_model, 3 * d_model),
                "proj": nn.Linear(d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            }))

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def hidden(self, ids):
        """(B, T) token ids -> (B, T, d_model) hidden states; position t sees tokens 0..t only."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]

        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)

        for block in self.blocks:
            h = block["ln1"](x)
            qkv = block["qkv"](h)
            q, k, v = qkv.split(self.d_model, dim=-1)
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, T, self.d_model)
            x = x + block["proj"](out)
            x = x + block["mlp"](block["ln2"](x))

        return self.ln_f(x)

    def forward(self, ids):
        """(B, T) token ids -> (B, T, vocab_size) next-token logits."""
        return self.lm_head(self.hidden(ids))
