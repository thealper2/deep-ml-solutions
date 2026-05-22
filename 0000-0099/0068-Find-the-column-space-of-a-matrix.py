import numpy as np

def matrix_image(A):
	A = np.array(A)
	Q, R = np.linalg.qr(A, mode='reduced')
	rank = np.linalg.matrix_rank(A)
	pivot_cols = []
	for j in range(R.shape[1]):
		if np.abs(R[j, j]) > 1e-10:
			pivot_cols.append(j)
		if len(pivot_cols) == rank:
			break

	result = A[:, pivot_cols]
	return result.tolist()