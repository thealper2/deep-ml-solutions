import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
	"""
	Compute the Jensen-Shannon Divergence between two probability distributions.
	
	Args:
		P: First probability distribution
		Q: Second probability distribution
	
	Returns:
		Jensen-Shannon Divergence value
	"""
	P = np.array(P, dtype=float)
	Q = np.array(Q, dtype=float)
	P = np.clip(P, 1e-12, 1)
	Q = np.clip(Q, 1e-12, 1)
	P = P / np.sum(P)
	Q = Q / np.sum(Q)
	M = 0.5 * (P + Q)
	kl_pm = np.sum(P * np.log(P / M))
	kl_qm = np.sum(Q * np.log(Q / M))
	jsd = 0.5 * kl_pm + 0.5 * kl_qm
	return round(jsd, 6)