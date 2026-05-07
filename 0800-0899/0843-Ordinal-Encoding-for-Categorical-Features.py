import numpy as np

def ordinal_encode(X, categories):
    """
    Encode categorical features as integer codes using a provided ordering.

    Args:
        X: 2D list/array of shape (n_samples, n_features) with string values.
        categories: list of length n_features; categories[j] is an ordered
                    list of valid categories for column j.

    Returns:
        np.ndarray of shape (n_samples, n_features), dtype int.
        Unknown categories are encoded as -1.
    """
    n_samples = len(X)
    n_features = len(categories)
    result = np.zeros((n_samples, n_features), dtype=int)

    for j in range(n_features):
        mapping = {cat: idx for idx, cat in enumerate(categories[j])}
        for i in range(n_samples):
            val = X[i][j]
            result[i, j] = mapping.get(val, -1)

    return result