import numpy as np

def prune_attention_heads(
    attention_weights: np.ndarray,
    head_importance_scores: np.ndarray,
    pruning_ratio: float
) -> tuple[np.ndarray, list[int]]:
	"""
	Prune less important attention heads to reduce inference cost.
	
	Transformer models have redundant heads. Research shows 50% of
	BERT heads can be removed with <1% accuracy loss.
	
	Args:
		attention_weights: Attention weights from all heads
		  Shape: (num_heads, seq_len, seq_len)
		head_importance_scores: Importance score per head
		  Shape: (num_heads,)
		  Higher score = more important
		pruning_ratio: Fraction of heads to prune
		  Range: 0.0 (keep all) to 1.0 (prune all)
	
	Returns:
		Tuple of (pruned_attention_weights, kept_head_indices):
		- pruned_attention_weights: Reduced attention matrices
		- kept_head_indices: List of preserved head indices
	"""
	num_heads = len(attention_weights)
	num_heads_to_keep = int(num_heads * (1 - pruning_ratio))

	if num_heads_to_keep == 0:
		return [], []

	head_indices = list(range(num_heads))
	sorted_heads = sorted(head_indices, key=lambda x: head_importance_scores[x], reverse=True)
	keep_indices = sorted(sorted_heads[:num_heads_to_keep])
	pruned_weights = np.array([attention_weights[i] for i in keep_indices])
	return pruned_weights, keep_indices