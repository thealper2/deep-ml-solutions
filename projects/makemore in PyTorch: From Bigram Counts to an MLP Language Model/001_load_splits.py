import os
import random
import tempfile
import urllib.request
import torch

def load_splits(block_size=8, seed=42):
    """Return (itos, {'train': (X, Y), 'dev': (X, Y), 'test': (X, Y)}) with block_size characters of context."""
    path = os.path.join(tempfile.gettempdir(), "makemore_names.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
            path,
        )
    with open(path, encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]

    chars = sorted(set(ch for w in names for ch in w))
    stoi = {'.': 0}
    for i, ch in enumerate(chars, start=1):
        stoi[ch] = i
    itos = {i: ch for ch, i in stoi.items()}

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    parts = {
        "train": shuffled[:n1],
        "dev": shuffled[n1:n2],
        "test": shuffled[n2:],
    }

    def build(words):
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
        return (
            torch.tensor(X, dtype=torch.long),
            torch.tensor(Y, dtype=torch.long),
        )

    splits = {k: build(v) for k, v in parts.items()}
    return itos, splits