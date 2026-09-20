import numpy as np
from itertools import product

def fourier_basis_features(s: np.ndarray, order: int) -> np.ndarray:
	"""
	Compute Fourier basis features for a given state.

	Args:
		s: State vector of shape (d,), each component in [0, 1]
		order: Non-negative integer specifying the Fourier basis order

	Returns:
		1D numpy array of (order+1)^d feature values, rounded to 4 decimals
	"""
	d = s.shape[0]
	n = order

	coeffs = list(product(range(n + 1), repeat=d))
	coeffs = [c for c in coeffs]
	coeffs.sort()

	features = []
	for c in coeffs:
		c_vec = np.array(c, dtype=float)
		features.append(np.cos(np.pi * np.dot(c_vec, s)))

	return np.round(np.array(features), 4)
