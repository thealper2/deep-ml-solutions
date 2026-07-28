def partition_indices(indices, train_ratio, val_ratio):
    N = len(indices)
    test_ratio = 1 - train_ratio - val_ratio
    train_size = int(train_ratio * N)
    val_size = int(val_ratio * N)
    test_size = int(test_ratio * N)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    return train_idx, val_idx, test_idx
