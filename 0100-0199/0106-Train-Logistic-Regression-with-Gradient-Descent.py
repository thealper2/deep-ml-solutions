import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	X_bias = np.c_[np.ones(X.shape[0]), X]
	n_samples, n_features = X_bias.shape

	weights = np.zeros(n_features)
	losses = []
	eps = 1e-12

	for _ in range(iterations):
		z = X_bias @ weights
		probs = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
		probs = np.clip(probs, eps, 1 - eps)
		loss = -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
		losses.append(round(loss, 4))
		gradient = X_bias.T @ (probs - y)
		weights -= learning_rate * gradient

	weights = np.round(weights, 4)
	return weights.tolist(), losses
