import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    A = np.array(matrix, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError()

    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_sorted = np.sort(eigenvalues)
    is_symmetric = np.allclose(A, A.T)
    is_positive_definite = np.all(eigenvalues.real > 1e-10)
    eigenvalues_rounded = []
    for val in eigenvalues_sorted:
        if abs(val.imag) < 1e-10:
            eigenvalues_rounded.append(round(val.real, 4))
        else:
            eigenvalues_rounded.append(round(val, 4))

    return {
        'is_positive_definite': is_positive_definite,
        'eigenvalues': eigenvalues_rounded,
    }