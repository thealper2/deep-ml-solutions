import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    M = np.array(M, dtype=np.float64)
    
    frob_norm = np.linalg.norm(M, 'fro')
    if frob_norm == 0:
        return M.tolist()
    
    M = M / frob_norm
    
    for _ in range(num_iters):
        MT = M.T
        MMT = M @ MT
        MMT2 = MMT @ MMT
        M = a * M + b * MMT @ M + c * MMT2 @ M
    
    return M.tolist()
