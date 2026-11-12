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