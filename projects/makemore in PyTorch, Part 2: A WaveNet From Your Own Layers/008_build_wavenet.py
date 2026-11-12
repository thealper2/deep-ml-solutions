import torch
import torch.nn.functional as F

def build_wavenet(vocab_size=27, n_embd=10, n_hidden=68, seed=42):
    """Embedding, three (FlattenConsecutive(2), Linear, BatchNorm1d, Tanh) blocks, then the output Linear."""
    g = torch.Generator()
    g.manual_seed(seed)
    layers = [
        Embedding(vocab_size, n_embd, generator=g),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size, generator=g),
    ]
    model = Sequential(layers)
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    for p in model.parameters():
        p.requires_grad = True
    return model