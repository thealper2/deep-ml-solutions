import torch
from torch.utils.data import DataLoader

def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_train = len(X_train)
    train_size = n_train - val_size

    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val_split = X_train[train_size:]
    y_val_split = y_train[train_size:]

    train_ds = FashionDataset(X_train_split, y_train_split)
    val_ds = FashionDataset(X_val_split, y_val_split)
    test_ds = FashionDataset(X_test, y_test)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "sizes": (len(train_ds), len(val_ds), len(test_ds))
    }