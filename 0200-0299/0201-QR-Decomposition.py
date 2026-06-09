import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	Q, R = np.linalg.qr(A)
    for i in range(min(R.shape)):
        if R[i, i] < 0:
            R[i, :] *= -1
            Q[:, i] *= -1

	return Q.tolist(), R.tolist()
