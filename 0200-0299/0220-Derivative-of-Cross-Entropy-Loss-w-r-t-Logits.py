import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	logits = np.array(logits)
	max_logit = np.max(logits)
	exp_logits = np.exp(logits - max_logit)
	probs = exp_logits / np.sum(exp_logits)
	one_hot = np.zeros_like(probs)
	one_hot[target] = 1.0
	gradient = probs - one_hot
	return gradient.tolist()