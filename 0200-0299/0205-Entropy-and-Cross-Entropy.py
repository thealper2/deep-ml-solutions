import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	P = np.clip(P, 1e-8, 1 - 1e-8)
	Q = np.clip(Q, 1e-8, 1 - 1e-8)
	entropy = -np.sum(P * np.log(P))
	cross_entropy = -np.sum(P * np.log(Q))
	return entropy, cross_entropy