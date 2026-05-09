import numpy as np

def mds(D: np.ndarray, k: int) -> np.ndarray:
    """
    Classical Multidimensional Scaling.

    Args:
        D: (n, n) symmetric pairwise distance matrix with zero diagonal.
        k: target embedding dimension.

    Returns:
        (n, k) embedding matrix.
    """
    n = D.shape[0]
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues_top = eigenvalues[:k]
    eigenvectors_top = eigenvectors[:, :k]
    eigenvalues_top = np.maximum(eigenvalues_top, 0)
    for i in range(k):
        vec = eigenvectors_top[:, i]
        for val in vec:
            if abs(val) > 1e-10:
                if val < 0:
                    eigenvectors_top[:, i] = -vec

                break

    embedding = eigenvectors_top @ np.diag(np.sqrt(eigenvalues_top))
    return embedding