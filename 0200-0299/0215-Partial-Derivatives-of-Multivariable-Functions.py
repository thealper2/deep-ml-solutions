import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.

	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate

	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	h = 1e-6
	point = np.array(point, dtype=float)

	def f(x):
		if func_name == 'poly2d':
			return x[0]**2 * x[1] + x[0] * x[1]**2
		elif func_name == 'exp_sum':
			return np.exp(x[0] + x[1])
		elif func_name == 'product_sin':
			return x[0] * np.sin(x[1])
		elif func_name == 'poly3d':
			return x[0]**2 * x[1] + x[1] * x[2]**2
		elif func_name == 'squared_error':
			return (x[0] - x[1])**2
		else:
			raise ValueError(f"Unknown function: {func_name}")

	n = len(point)
	grad = np.zeros(n)

	for i in range(n):
		point_plus = point.copy()
		point_minus = point.copy()
		point_plus[i] += h
		point_minus[i] -= h
		grad[i] = (f(point_plus) - f(point_minus)) / (2 * h)

	return tuple(grad)