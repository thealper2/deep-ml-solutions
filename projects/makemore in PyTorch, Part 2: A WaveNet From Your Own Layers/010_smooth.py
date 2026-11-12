import torch

def smooth(values, k):
    """Mean of every k consecutive values; ValueError unless len(values) is a multiple of k."""
    if len(values) % k != 0:
        raise ValueError("length is not a multiple of k")
    out = []
    for i in range(0, len(values), k):
        chunk = values[i:i + k]
        out.append(sum(chunk) / k)
    return out