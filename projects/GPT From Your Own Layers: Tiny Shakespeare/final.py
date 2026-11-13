"""
GPT From Your Own Layers: Tiny Shakespeare — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_data ──
import os
import tempfile
import urllib.request
import torch

def load_data(train_frac=0.9):
    """Return (chars, encode, decode, train_data, val_data) for Tiny Shakespeare, one id per character."""
    path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return chars, encode, decode, data[:n], data[n:]

# ── Step 002  get_batch ──
import torch

def get_batch(data, block_size, batch_size, generator=None):
    """Random (batch_size, block_size) chunks x of data and their next-token targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# ── Step 003  BigramLanguageModel ──
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """One embedding table: row i holds the logits for the token after token i."""
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Return (logits of shape (B, T, vocab_size), loss or None)."""
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))
        return logits, loss

# ── Step 004  generate ──
import torch
import torch.nn.functional as F

def generate(model, idx, max_new_tokens, block_size, generator=None):
    """Append max_new_tokens sampled tokens to idx (B, T), cropping the context to block_size."""
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_token], dim=1)
    return idx

# ── Step 005  train ──
import torch

def train(model, data, block_size, batch_size, steps, lr=1e-3, generator=None):
    """AdamW training on random batches; returns each step's loss as a float."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        xb, yb = get_batch(data, block_size, batch_size, generator=generator)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses

# ── Step 006  estimate_loss ──
import torch

def estimate_loss(model, splits, block_size, batch_size, eval_iters, generator=None):
    """Mean loss over eval_iters batches for each named split, in evaluation mode, as floats."""
    model.eval()
    out = {}
    with torch.no_grad():
        for name, data in splits.items():
            total = 0.0
            for _ in range(eval_iters):
                xb, yb = get_batch(data, block_size, batch_size, generator=generator)
                _, loss = model(xb, yb)
                total += float(loss.item())
            out[name] = total / eval_iters
    model.train()
    return out

# ── Step 007  average_past ──
import torch
import torch.nn.functional as F

def average_past(x):
    """Position t of the output is the mean of x[:, :t + 1], computed with a masked softmax and a matrix multiply."""
    B, T, C = x.shape
    w = torch.zeros((T, T))
    w = w.masked_fill(torch.triu(torch.ones((T, T)), diagonal=1).bool(), float('-inf'))
    w = F.softmax(w, dim=-1)
    return w @ x

# ── Step 008  Head ──
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

# ── Step 009  MultiHeadAttention ──
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

# ── Step 010  FeedForward ──
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        """Linear(n_embd, 4 n_embd) -> ReLU -> Linear(4 n_embd, n_embd) -> Dropout, as self.net."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ── Step 011  LayerNorm ──
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """Learnable gamma (ones) and beta (zeros) of shape (dim,)."""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """Normalize each vector along the last dimension, then scale and shift."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta

# ── Step 012  Block ──
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

# ── Step 013  GPTLanguageModel ──
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        """Token and position embeddings, n_layer Blocks, a final LayerNorm and a linear head."""
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Return (logits of shape (B, T, vocab_size), loss or None), like the bigram model."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))
        return logits, loss

# ── Scaffold (runner) ──
"""GPT from your own layers, on Tiny Shakespeare.

Story: the bigram baseline from the start of the video, then a small GPT built from your Head,
MultiHeadAttention, FeedForward, LayerNorm and Block, trained on the same data with the same budget.
Compare their validation losses and read what each one writes.
(The video's full model, n_embd 384, 6 heads, 6 layers, block size 256, needs a GPU: about 1.48.)
"""
import torch


def main():
    chars, encode, decode, train_data, val_data = load_data()
    splits = {"train": train_data, "val": val_data}
    print(f"{len(train_data) + len(val_data):,} characters, vocabulary of {len(chars)}")

    # ---- 1. The bigram baseline ----
    torch.manual_seed(1337)
    bigram = BigramLanguageModel(len(chars))
    train(bigram, train_data, block_size=8, batch_size=32, steps=1500, lr=1e-2, generator=torch.Generator().manual_seed(0))
    est = estimate_loss(bigram, splits, 8, 32, 50, generator=torch.Generator().manual_seed(1))
    print("bigram  train {train:.3f}  val {val:.3f}".format(**est))

    # ---- 2. A small GPT ----
    block_size = 32
    torch.manual_seed(1337)
    gpt = GPTLanguageModel(len(chars), n_embd=64, n_head=4, n_layer=3, block_size=block_size, dropout=0.0)
    print(f"GPT parameters: {sum(p.numel() for p in gpt.parameters()):,}")
    train(gpt, train_data, block_size, batch_size=32, steps=1500, lr=2e-3, generator=torch.Generator().manual_seed(0))
    est = estimate_loss(gpt, splits, block_size, 32, 50, generator=torch.Generator().manual_seed(1))
    print("GPT     train {train:.3f}  val {val:.3f}".format(**est))

    # ---- 3. What they write ----
    start = torch.zeros((1, 1), dtype=torch.long)
    print("--- bigram ---")
    print(decode(generate(bigram, start, 200, 8, generator=torch.Generator().manual_seed(7))[0].tolist()))
    print("--- GPT ---")
    print(decode(generate(gpt, start, 300, block_size, generator=torch.Generator().manual_seed(7))[0].tolist()))


if __name__ == "__main__":
    main()
