import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    n_samples, n_features = X_minority.shape
    k_actual = min(k, n_samples - 1)

    if k_actual == 0 or n_synthetic == 0:
        return np.zeros((0, n_features))

    synthetic = np.zeros((n_synthetic, n_features))

    for idx in range(n_synthetic):
        i = np.random.randint(0, n_samples)
        x_i = X_minority[i]
        distances = np.linalg.norm(X_minority - x_i, axis=1)
        distances[i] = np.inf
        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_neighbors = X_minority[nearest_indices]
        j = np.random.randint(0, k_actual)
        x_nn = nearest_neighbors[j]
        gap = np.random.random()
        synthetic[idx] = x_i + gap * (x_nn - x_i)

    return synthetic
