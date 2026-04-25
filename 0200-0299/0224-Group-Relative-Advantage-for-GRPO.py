import numpy as np

def compute_group_relative_advantage(rewards: list[float]) -> list[float]:
	"""
	Compute the Group Relative Advantage for GRPO.
	
	For each reward r_i in a group, compute:
	A_i = (r_i - mean(rewards)) / std(rewards)
	
	If all rewards are identical (std=0), return zeros.
	
	Args:
		rewards: List of rewards for a group of outputs from the same prompt
		
	Returns:
		List of normalized advantages
	"""
	mean = np.mean(rewards)
	std = np.std(rewards)
	normalized = [(x - mean) / std if std else 0.0 for x in rewards]
	return normalized