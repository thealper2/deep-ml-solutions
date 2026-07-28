def one_hot_encode(labels):
    unique = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique)}
    N = len(labels)
    C = len(unique)
    one_hot = np.zeros((N, C), dtype=float)
    for i, label in enumerate(labels):
        one_hot[i, label_to_idx[label]] = 1.0

    return one_hot
