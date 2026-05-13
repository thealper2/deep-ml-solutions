import numpy as np

def bitfit_select(params: dict) -> dict:
	"""
	Select bias-only parameters for BitFit fine-tuning.

	Args:
		params: dict mapping parameter name (str) to numpy ndarray

	Returns:
		dict mapping each parameter name to a bool (True if trainable),
		plus a key 'trainable_ratio' (float) giving the fraction of
		total scalar params that are trainable, rounded to 6 decimals.
	"""
	result = {}
	total_params = 0
	trainable_params = 0

	for name, array in params.items():
		total_params += array.size
		if 'bias' in name:
			result[name] = True
			trainable_params += array.size
		else:
			result[name] = False

	result['trainable_ratio'] = round(trainable_params / total_params, 6)
	return result