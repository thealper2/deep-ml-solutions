import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    A = np.array(list(A))
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)

    eigenvalues_V, V = np.linalg.eigh(AtA)
    eigenvalues_U, U = np.linalg.eigh(AAt)

    idx_V = np.argsort(eigenvalues_V)[::-1]
    idx_U = np.argsort(eigenvalues_U)[::-1]

    V = V[:, idx_V]
    U = U[:, idx_U]
    
    singular_values = np.sqrt(np.abs(eigenvalues_V[idx_V]))
    print(U @ np.diag(singular_values) @ V.T)
    return U.tolist(), singular_values.tolist(), V.T.tolist()

A = [[-10, 8],
	 [10, -1]]

U, Sigma, Vt = svd_2x2(A)
print(U)
print(Sigma)
print(Vt)