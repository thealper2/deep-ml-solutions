import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	C = np.max(y) + 1
	n_samples, n_features = X.shape

	X_bias = np.c_[np.ones(n_samples), X]
	M = n_features + 1

	W = np.zeros((C, M))

	y_one_hot = np.zeros((n_samples, C))
	y_one_hot[np.arange(n_samples), y] = 1

	losses = []

	for _ in range(iterations):
		logits = X_bias @ W.T

		max_logits = np.max(logits, axis=1, keepdims=True)
		exp_logits = np.exp(logits - max_logits)
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

		epsilon = 1e-12
		probs = np.clip(probs, epsilon, 1 - epsilon)
		loss = -np.sum(y_one_hot * np.log(probs))
		losses.append(round(loss, 4))

		gradient = (probs - y_one_hot).T @ X_bias

		W -= learning_rate * gradient

	coefficients = [row.tolist() for row in W]
	return coefficients, losses
