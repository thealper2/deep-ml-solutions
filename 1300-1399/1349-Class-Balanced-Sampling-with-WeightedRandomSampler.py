import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels, num_samples):
    counts = torch.bincount(labels)
    counts_float = counts.float()
    weights = 1.0 / counts_float[labels]
    sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
    return sampler