import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	d_k = Q.shape[-1]
	scores = np.matmul(Q, K.T) / np.sqrt(d_k)
	if mask is not None:
		scores = np.where(mask == 0, -np.inf, scores)

	max_vals = np.max(scores, axis=-1, keepdims=True)
	exp_scores = np.exp(scores - max_vals)
	weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
	output = np.matmul(weights, V)
	return output.tolist(), weights.tolist()