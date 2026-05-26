import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    m, n = A.shape
    U, s, Vh = np.linalg.svd(A)
    rank = np.sum(s > tol)
    nullity = n - rank

    if nullity == 0:
        return np.empty((n, 0))

    null_space_basis = Vh[rank:, :].T

    for i in range(nullity):
        norm = np.linalg.norm(null_space_basis[:, i])
        if norm > tol:
            null_space_basis[:, i] = null_space_basis[:, i] / norm

    return null_space_basis