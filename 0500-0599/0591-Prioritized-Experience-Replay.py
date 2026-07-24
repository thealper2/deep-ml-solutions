import numpy as np

def prioritized_replay_sample(priorities: list, batch_size: int, alpha: float = 0.6, beta: float = 0.4, seed: int = 42) -> dict:
	"""
	Sample a batch from a replay buffer using prioritized experience replay.

	Args:
		priorities: list of priority values for each experience (positive floats)
		batch_size: number of experiences to sample
		alpha: prioritization exponent (0 = uniform, 1 = full prioritization)
		beta: importance sampling exponent (0 = no correction, 1 = full correction)
		seed: random seed for reproducibility

	Returns:
		dict with 'indices', 'probabilities', and 'weights'
	"""
	priorities = np.array(priorities)
	N = len(priorities)
	probs = (priorities ** alpha) / np.sum(priorities ** alpha)
	rng = np.random.RandomState(seed) if seed is not None else np.random
	indices = rng.choice(N, size=batch_size, replace=False, p=probs)
	weights = (N * probs[indices]) ** (-beta)
	weights = weights / np.max(weights)

	return {
		'indices': indices.tolist(),
		'probabilities': np.round(probs, 4).tolist(),
		'weights': np.round(weights, 4).tolist(),
	}
