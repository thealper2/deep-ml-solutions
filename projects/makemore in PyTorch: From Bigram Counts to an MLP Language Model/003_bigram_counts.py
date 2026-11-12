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