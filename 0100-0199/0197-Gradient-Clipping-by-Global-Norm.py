import numpy as np

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
	"""
	Clip gradients by global norm.
	
	Args:
		gradients: List of gradient arrays
		max_norm: Maximum allowed global norm
	
	Returns:
		List of clipped gradient arrays
	"""
	flat_grads = []
	for grad in gradients:
		flat_grads.extend(grad)

	flat_grads = np.array(flat_grads)
	global_norm = np.sqrt(np.sum(flat_grads ** 2))

	if global_norm <= max_norm:
		return gradients

	scaling_factor = max_norm / global_norm

	clipped = []
	for grad in gradients:
		grad_array = np.array(grad)
		grad_scaled = grad_array * scaling_factor
		clipped.append(grad_scaled.tolist())

	return clipped