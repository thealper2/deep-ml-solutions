import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    return torch.full(shape, epsilon / (vocab_size - 2), dtype=torch.float32)
