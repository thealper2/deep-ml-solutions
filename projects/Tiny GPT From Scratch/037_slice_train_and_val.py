def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    train = data[:split_idx]
    val = data[split_idx:]
    return train, val
