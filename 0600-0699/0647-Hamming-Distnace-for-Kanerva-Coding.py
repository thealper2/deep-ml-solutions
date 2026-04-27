import numpy as np

def hamming_distance_kanerva(state: list, prototypes: list, threshold: int) -> tuple:
	"""
	Compute Hamming distances and find active prototypes for Kanerva coding.

	Args:
		state: Binary state vector (list of 0s and 1s).
		prototypes: List of binary prototype vectors.
		threshold: Maximum Hamming distance for a prototype to be active.

	Returns:
		Tuple of (distances, active_indices).
	"""
	state = np.array(state)
	prototypes = np.array(prototypes)
	distances = np.sum(state != prototypes, axis=1).tolist()
	active_indices = [i for i, d in enumerate(distances) if d <= threshold]
	return distances, active_indices