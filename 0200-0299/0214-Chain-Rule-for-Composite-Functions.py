import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	def derivative(f, value):
		if f == 'square':
			return 2 * value
		elif f == 'sin':
			return np.cos(value)
		elif f == 'exp':
			return np.exp(value)
		elif f == 'log':
			return 1.0 / value
		else:
			return 1.0

	grad = 1.0
	current_x = x

	for f in functions[::-1]:
		grad *= derivative(f, current_x)
		if f == 'square':
			current_x = current_x ** 2
		elif f == 'sin':
			current_x = np.sin(current_x)
		elif f == 'exp':
			current_x = np.exp(current_x)
		elif f == 'log':
			current_x = np.log(current_x)

	return round(grad, 4)