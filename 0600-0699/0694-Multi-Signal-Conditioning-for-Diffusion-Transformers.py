import numpy as np

def multi_signal_conditioning(features: np.ndarray, signals: list, proj_weights: list, proj_biases: list, mod_weight: np.ndarray, mod_bias: np.ndarray) -> np.ndarray:
	"""
	Modulate hidden features using multiple conditioning signals.

	Args:
		features: (N, D) hidden feature array
		signals: list of K conditioning signal arrays
		proj_weights: list of K projection weight matrices
		proj_biases: list of K projection bias vectors
		mod_weight: modulation weight matrix
		mod_bias: modulation bias vector

	Returns:
		(N, D) modulated features
	"""
	N, D = features.shape
	combined = None
	for s, W, b in zip(signals, proj_weights, proj_biases):
		proj = s @ W + b
		if combined is None:
			combined = proj
		else:
			combined += proj

	params = combined @ mod_weight + mod_bias
	gamma, beta = params[:, :D], params[:, D:]
	output = (1.0 + gamma) * features + beta
	return output