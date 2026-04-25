import numpy as np

def kl_divergence_estimator(pi_theta: np.ndarray, pi_ref: np.ndarray) -> np.ndarray:
	"""
	Compute the unbiased KL divergence estimator used in GRPO.
	
	Formula: D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
	
	Args:
		pi_theta: Current policy probabilities for each sample
		pi_ref: Reference policy probabilities for each sample
		
	Returns:
		Array of KL divergence estimates (one per sample)
	"""
	d_kl = (pi_ref / pi_theta) - np.log(pi_ref / pi_theta) - 1
	return d_kl