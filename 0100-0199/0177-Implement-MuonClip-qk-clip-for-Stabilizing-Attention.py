import numpy as np

def muonclip_qk_clip(W_q: np.ndarray, W_k: np.ndarray, x: np.ndarray, t: float, alpha: float = 0.5, eps: float = 1e-7):
	"""
	Apply MuonClip qk-clip to (W_q, W_k).

	Args:
		W_q: (d_head, d_model) query projection weights
		W_k: (d_head, d_model) key  projection weights
		x: (batch, seq, d_model) input features
		t: threshold for max QK score (after 1/sqrt(d_head) scaling)
		alpha: fraction of rescaling applied to W_q (remainder to W_k)
		eps: small epsilon to avoid division by zero

	Returns:
		W_q_new (list[list[float]]), W_k_new (list[list[float]]), clipped (bool), max_post (float rounded to 4 dp)
	"""
	W_q = np.array(W_q)
	W_k = np.array(W_k)
	x = np.array(x)

	d_head = W_q.shape[0]
	scale = 1.0 / np.sqrt(d_head)

	Q = x @ W_q.T
	K = x @ W_k.T

	scores = (Q @ K.transpose(0, 2, 1)) * scale

	max_score = np.max(scores)

	if max_score <= t:
		clipped = False
		max_post = max_score
		W_q_new = W_q
		W_k_new = W_k
	else:
		clipped = True
		eta = t / max_score
		eta_q = eta ** alpha
		eta_k = eta ** (1 - alpha)
		W_q_new = W_q * eta_q
		W_k_new = W_k * eta_k
		max_post = t

	W_q_new = np.round(W_q_new, 4).tolist()
	W_k_new = np.round(W_k_new, 4).tolist()
	max_post = round(max_post, 4)

	return W_q_new, W_k_new, clipped, max_post
