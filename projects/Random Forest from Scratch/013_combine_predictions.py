def combine_predictions(tree_predictions):
    n_trees, n_samples = tree_predictions.shape
    result = np.zeros(n_samples, dtype=int)

    for j in range(n_samples):
        votes = tree_predictions[:, j]
        counts = np.bincount(votes)
        result[j] = np.argmax(counts)

    return result
