import numpy as np

def tsne_gradient(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
	"""
	Compute the gradient of the t-SNE cost function.
	
	Args:
		P: (n, n) symmetric numpy array of joint probabilities in high-dimensional space
		Y: (n, d) numpy array of current low-dimensional embedding
	
	Returns:
		gradient: (n, d) numpy array of gradients for each point, rounded to 4 decimal places
	"""
	P = np.array(P)
	Y = np.array(Y)
	n, d = Y.shape

	diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
	sq_dist = np.sum(diff ** 2, axis=2)

	Q_numer = 1.0 / (1.0 + sq_dist)
	np.fill_diagonal(Q_numer, 0.0)
	Q = Q_numer / np.sum(Q_numer)

	grad = np.zeros_like(Y)

	factor = (P - Q) * Q_numer

	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			
			grad[i] += factor[i, j] * (Y[i] - Y[j])

	grad = 4.0 * grad
	return np.round(grad, 4)