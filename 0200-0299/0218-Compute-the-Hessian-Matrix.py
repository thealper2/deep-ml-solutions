from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	n = len(point)
	H = np.zeros((n, n))

	for i in range(n):
		for j in range(n):
			x_pp = point.copy()
			x_pm = point.copy()
			x_mp = point.copy()
			x_mm = point.copy()

			x_pp[i] += h 
			x_pp[j] += h
			x_pm[i] += h
			x_pm[j] -= h
			x_mp[i] -= h
			x_mp[j] += h
			x_mm[i] -= h
			x_mm[j] -= h

			H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h * h)
        
	return np.round(H, 4)