def standardize_and_add_bias(splits):
    X_train = splits['X_train']
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)
    std[std == 0] = 1.0

    std_splits = {}
    for key, value in splits.items():
        if key.startswith('X_'):
            X_std = (value - mean) / std
            std_splits[key] = add_bias_column(X_std)
        else:
            std_splits[key] = value

    return std_splits, mean, std
