import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size=64, seed=0, shuffle=True):
    """DataLoader over float32 X and int64 y, shuffled by its own seeded generator."""
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.int64)
    ds = TensorDataset(X_t, y_t)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)