import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    hessian = np.array(hessian)
    eigenvalues = np.linalg.eigvalsh(hessian) 

    if np.any(np.abs(eigenvalues) < tol):
        return None

    if np.all(eigenvalues > 0):
        return -1

    elif np.all(eigenvalues < 0):
        return 1
    
    else:
        return 0
