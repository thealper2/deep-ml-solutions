def bootstrap_sample(features, labels, rng):
    n = len(features)
    indices = rng.integers(0, n, size=n)
    return features[indices], labels[indices]
