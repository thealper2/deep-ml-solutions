def train_test_split(x, y, test_fraction=0.2, seed=0):
    N = len(x)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(N)
    split_idx = int(N * test_fraction)
    train_indices = shuffled_indices[split_idx:]
    test_indices = shuffled_indices[:split_idx]

    tr_f = x[train_indices]
    tr_l = y[train_indices]
    te_f = x[test_indices]
    te_l = y[test_indices]

    return tr_f, tr_l, te_f, te_l
