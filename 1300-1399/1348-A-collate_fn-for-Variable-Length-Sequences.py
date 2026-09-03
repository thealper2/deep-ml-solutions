import torch
import torch.nn as nn


def pad_collate(batch, pad_value=0):
    sequences, labels = zip(*batch)
    T_max = max(seq.size(0) for seq in sequences)
    B = len(batch)
    dtype = sequences[0].dtype
    device = sequences[0].device
    padded = torch.full((B, T_max), pad_value, dtype=dtype, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)

    for i, seq in enumerate(sequences):
        L = seq.size(0)
        lengths[i] = L
        padded[i, :L] = seq

    mask = torch.arange(T_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return padded, lengths, mask, labels_tensor
