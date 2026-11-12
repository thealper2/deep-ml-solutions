"""
makemore in PyTorch: From Bigram Counts to an MLP Language Model — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_names ──
import os
import tempfile
import urllib.request

def load_names():
    """Cached download of makemore's names.txt; returns the list of names."""
    path = os.path.join(tempfile.gettempdir(), "makemore_names.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
            path,
        )
    with open(path, encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

# ── Step 002  build_vocab ──
def build_vocab(words):
    """Return (stoi, itos): '.' -> 0, the sorted characters -> 1, 2, ..."""
    chars = sorted(set(ch for w in words for ch in w))
    stoi = {'.': 0}
    for i, ch in enumerate(chars, start=1):
        stoi[ch] = i
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

# ── Step 003  bigram_counts ──
import torch

def bigram_counts(words, stoi):
    """(V, V) int32 tensor of bigram counts, with '.' around every word."""
    V = len(stoi)
    N = torch.zeros((V, V), dtype=torch.int32)
    for w in words:
        s = [0] + [stoi[c] for c in w] + [0]
        for a, b in zip(s, s[1:]):
            N[a, b] += 1

    return N

# ── Step 004  bigram_probs ──
def bigram_probs(N, smoothing=1):
    """Row-normalized float probabilities after adding `smoothing` to every count."""
    N = N.float() + smoothing
    return N / N.sum(dim=1, keepdim=True)

# ── Step 005  sample_bigram ──
import torch

def sample_bigram(P, itos, n, seed):
    """Sample n names, one generator seeded with `seed` for every draw."""
    g = torch.Generator()
    g.manual_seed(seed)
    names = []
    for _ in range(n):
        cur = 0
        chars = []
        while True:
            nxt = int(torch.multinomial(P[cur], 1, replacement=True, generator=g).item())
            if nxt == 0:
                break
            chars.append(itos[nxt])
            cur = nxt
        names.append("".join(chars))
    return names

# ── Step 006  bigram_nll ──
import torch
import math

def bigram_nll(P, words, stoi):
    """Average negative log-likelihood per bigram, as a float."""
    total = 0.0
    count = 0
    for w in words:
        s = [0] + [stoi[c] for c in w] + [0]
        for a, b in zip(s, s[1:]):
            total -= math.log(float(P[a, b]))
            count += 1
    return total / count

# ── Step 007  bigram_dataset ──
import torch

def bigram_dataset(words, stoi):
    """Return (xs, ys): input and next-token ids for every bigram, as long tensors."""
    xs = []
    ys = []
    for w in words:
        s = [0] + [stoi[c] for c in w] + [0]
        for a, b in zip(s, s[1:]):
            xs.append(a)
            ys.append(b)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

# ── Step 008  train_neural_bigram ──
import torch
import torch.nn.functional as F

def train_neural_bigram(xs, ys, vocab_size, steps, lr, reg, seed):
    """Full-batch gradient descent on one linear layer; returns (W, losses)."""
    g = torch.Generator()
    g.manual_seed(seed)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    losses = []
    n = len(xs)
    for _ in range(steps):
        x_onehot = torch.zeros(n, vocab_size)
        x_onehot[torch.arange(n), xs] = 1.0
        logits = x_onehot @ W
        probs = torch.softmax(logits, dim=1)
        nll = -torch.log(probs[torch.arange(n), ys]).mean()
        loss = nll + reg * (W ** 2).mean()
        losses.append(float(loss.item()))
        if W.grad is not None:
            W.grad.zero_()
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
    return W, losses

# ── Step 009  build_context_dataset ──
import torch

def build_context_dataset(words, stoi, block_size):
    """Return (X, Y): '.'-padded rolling contexts of length block_size and the next-token ids."""
    X = []
    Y = []
    for w in words:
        ctx = [0] * block_size
        for ch in w:
            cid = stoi[ch]
            X.append(list(ctx))
            Y.append(cid)
            ctx = ctx[1:] + [cid]
        X.append(list(ctx))
        Y.append(0)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# ── Step 010  split_words ──
import random

def split_words(words, seed):
    """Shuffle a copy with random.Random(seed); return the 80/10/10 (train, dev, test) split."""
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]

# ── Step 011  MLP ──
import torch

class MLP:
    def __init__(self, vocab_size, block_size, emb_dim, hidden, seed):
        """Draw C, W1, b1, W2, b2 (in that order) from one seeded generator; all require grad."""
        self.block_size = block_size
        g = torch.Generator()
        g.manual_seed(seed)
        self.C = torch.randn((vocab_size, emb_dim), generator=g, requires_grad=True)
        self.W1 = torch.randn((block_size * emb_dim, hidden), generator=g, requires_grad=True)
        self.b1 = torch.randn((hidden,), generator=g, requires_grad=True)
        self.W2 = torch.randn((hidden, vocab_size), generator=g, requires_grad=True)
        self.b2 = torch.randn((vocab_size,), generator=g, requires_grad=True)

    def parameters(self):
        """[C, W1, b1, W2, b2]"""
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        """(n, block_size) ids -> (n, vocab_size) logits."""
        emb = self.C[X]
        flat = emb.reshape(X.shape[0], -1)
        h = torch.tanh(flat @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

# ── Step 012  evaluate_loss ──
import torch
import torch.nn.functional as F

def evaluate_loss(model, X, Y):
    """Mean cross-entropy of model(X) against Y, as a float, without gradients."""
    with torch.no_grad():
        logits = model(X)
        return float(F.cross_entropy(logits, Y).item())

# ── Step 013  train_mlp ──
import torch
import torch.nn.functional as F

def train_mlp(model, X, Y, steps, batch_size, lr, seed):
    """Minibatch gradient descent with lr, then lr / 10 for the second half; returns the losses."""
    g = torch.Generator()
    g.manual_seed(seed)
    n = X.shape[0]
    losses = []
    params = model.parameters()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        for p in params:
            p.grad = None
        loss.backward()
        cur_lr = lr if step < steps // 2 else lr / 10
        with torch.no_grad():
            for p in params:
                p -= cur_lr * p.grad
        losses.append(float(loss.item()))
    return losses

# ── Step 014  sample_mlp ──
import torch
import torch.nn.functional as F

def sample_mlp(model, itos, n, seed):
    """Sample n names from the MLP, starting each from an all-'.' context."""
    g = torch.Generator()
    g.manual_seed(seed)
    bs = model.block_size
    names = []
    with torch.no_grad():
        for _ in range(n):
            ctx = [0] * bs
            chars = []
            while True:
                X = torch.tensor([ctx], dtype=torch.long)
                logits = model(X)[0]
                probs = F.softmax(logits, dim=0)
                nxt = int(torch.multinomial(probs, 1, generator=g).item())
                if nxt == 0:
                    break
                chars.append(itos[nxt])
                ctx = ctx[1:] + [nxt]
            names.append("".join(chars))
    return names

# ── Scaffold (runner) ──
"""makemore in PyTorch: from bigram counts to an MLP language model.

Story: count bigrams over 32,000 names and sample from the table; learn the same table with
a one-layer network; then train an MLP that sees three characters, check it on names it never
trained on, and sample new names from it.
"""
import torch


def main():
    words = load_names()
    stoi, itos = build_vocab(words)

    # ---- 1. Counting ----
    N = bigram_counts(words, stoi)
    P = bigram_probs(N, smoothing=1)
    print("bigram loss on all names:", round(bigram_nll(P, words, stoi), 4))
    print("bigram samples:", sample_bigram(P, itos, n=8, seed=2147483647))

    # ---- 2. The same model as a network ----
    xs, ys = bigram_dataset(words, stoi)
    W, losses = train_neural_bigram(xs, ys, len(stoi), steps=100, lr=50.0, reg=0.01, seed=2147483647)
    print("neural bigram loss:", round(losses[0], 4), "->", round(losses[-1], 4))

    # ---- 3. An MLP over three characters ----
    train, dev, test = split_words(words, seed=42)
    Xtr, Ytr = build_context_dataset(train, stoi, block_size=3)
    Xdev, Ydev = build_context_dataset(dev, stoi, block_size=3)
    Xte, Yte = build_context_dataset(test, stoi, block_size=3)
    model = MLP(len(stoi), block_size=3, emb_dim=10, hidden=200, seed=2147483647)
    print("MLP parameters:", sum(p.nelement() for p in model.parameters()))
    train_mlp(model, Xtr, Ytr, steps=30000, batch_size=64, lr=0.1, seed=0)
    print("MLP train loss:", round(evaluate_loss(model, Xtr, Ytr), 4))
    print("MLP dev loss:  ", round(evaluate_loss(model, Xdev, Ydev), 4))
    print("MLP test loss: ", round(evaluate_loss(model, Xte, Yte), 4), "(checked once, at the end)")
    print("MLP samples:", sample_mlp(model, itos, n=8, seed=2147483647))


if __name__ == "__main__":
    main()
