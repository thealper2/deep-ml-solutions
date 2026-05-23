import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	x = np.array(x, dtype=float)
	n = len(x)

	f_x = np.array(f(x))
	m = len(f_x)

	J = np.zeros((m, n))
	for j in range(n):
		e = np.zeros(n)
		e[j] = h
		f_plus = np.array(f(x + e))
		J[:, j] = (f_plus - f_x) / h

	return J