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