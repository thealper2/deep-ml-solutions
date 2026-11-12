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