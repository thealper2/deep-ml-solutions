import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    first_batch_X = next(iter(dataloader))[0]
    return len(dataloader), tuple(first_batch_X.shape)
