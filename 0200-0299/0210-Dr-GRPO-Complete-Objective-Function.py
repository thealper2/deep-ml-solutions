import numpy as np

def compute_dr_grpo_objective(log_probs_new: list[list[float]], 
                               log_probs_old: list[list[float]], 
                               rewards: list[float], 
                               epsilon: float = 0.2) -> float:
	"""
	Compute the Dr. GRPO (GRPO Done Right) clipped objective.
	
	Args:
		log_probs_new: Log probabilities from new policy π_θ
		              Each response: [log π_θ(o_1|q), log π_θ(o_2|q,o_1), ...]
		log_probs_old: Log probabilities from old policy π_θ_old
		rewards: Rewards R(q, o_i) for each response
		epsilon: Clipping parameter for importance ratios
	
	Returns:
		Dr. GRPO objective value
	"""
	log_probs_new = np.array(log_probs_new)
	log_probs_old = np.array(log_probs_old)
	rewards = np.array(rewards)

	G = len(rewards)
	T = log_probs_new.shape[1]

	mean_r = np.mean(rewards)
	advantages = (rewards - mean_r).reshape(G, 1)

	ratio = np.exp(log_probs_new - log_probs_old)

	surr1 = ratio * advantages
	surr2 = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
	clipped_obj = np.minimum(surr1, surr2)

	per_response_obj  = np.sum(clipped_obj, axis=1)
	objective = np.mean(per_response_obj)
	return float(objective)