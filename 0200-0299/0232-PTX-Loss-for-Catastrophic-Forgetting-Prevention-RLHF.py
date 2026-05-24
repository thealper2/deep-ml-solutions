import numpy as np

def compute_ptx_loss(
    rl_loss: float,
    pretrain_logits: np.ndarray,
    pretrain_labels: np.ndarray,
    beta_ptx: float = 0.1
) -> tuple[float, float, float]:
	"""
	Compute PTX (Pre-training) Loss to prevent catastrophic forgetting in RLHF.
	
	PTX Loss = RL Loss + beta_ptx * Cross-Entropy Loss
	
	Prevents model from forgetting general capabilities while
	fine-tuning with reinforcement learning from human feedback.
	
	Args:
		rl_loss: Reinforcement learning loss (e.g., PPO objective)
		pretrain_logits: Model logits on pre-training batch
		  Shape: (batch_size, vocab_size)
		pretrain_labels: True token indices
		  Shape: (batch_size,)
		beta_ptx: Weight coefficient (typically 0.05-0.2)
	
	Returns:
		Tuple of (total_loss, ce_loss, weighted_ce_loss):
		- total_loss: L_RL + beta_ptx * L_CE
		- ce_loss: Cross-entropy on pre-training data
		- weighted_ce_loss: beta_ptx * L_CE
	"""
	logits = np.array(pretrain_logits)
	labels = np.array(pretrain_labels)

	max_logits = np.max(logits, axis=1, keepdims=True)
	exp_logits = np.exp(logits - max_logits)
	probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

	n_samples = len(labels)
	ce_loss = 0.0
	for i in range(n_samples):
		ce_loss += -np.log(probs[i, labels[i]] + 1e-12)

	ce_loss /= n_samples
	weighted_ce_loss = beta_ptx * ce_loss
	total_loss = rl_loss + weighted_ce_loss
	return total_loss, ce_loss, weighted_ce_loss
