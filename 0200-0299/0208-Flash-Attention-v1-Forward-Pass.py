import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                           block_size: int = 2) -> np.ndarray:
	"""
	Compute attention output using Flash Attention v1 algorithm.
	
	Args:
		Q: Query matrix (seq_len, d_model)
		K: Key matrix (seq_len, d_model)
		V: Value matrix (seq_len, d_model)
		block_size: Size of blocks for tiled computation
	
	Returns:
		Output matrix (seq_len, d_model)
	"""
	seq_len, d_model = Q.shape
	scale = 1.0 / np.sqrt(d_model)
	output = np.zeros((seq_len, d_model), dtype=Q.dtype)

	for i in range(0, seq_len, block_size):
		Qi = Q[i:i + block_size]
		bq = Qi.shape[0]

		m_i = np.full((bq, 1), -np.inf, dtype=Q.dtype)
		l_i = np.zeros((bq, 1), dtype=Q.dtype)
		O_i = np.zeros((bq, d_model), dtype=Q.dtype)

		for j in range(0, seq_len, block_size):
			Kj = K[j:j + block_size]
			Vj = V[j:j + block_size]

			S_ij = (Qi @ Kj.T) * scale

			m_new = np.maximum(m_i, np.max(S_ij, axis=1, keepdims=True))
			P_ij = np.exp(S_ij - m_new)
			alpha = np.exp(m_i - m_new)

			l_i = alpha * l_i + np.sum(P_ij, axis=1, keepdims=True)
			O_i = alpha * O_i + P_ij @ Vj

			m_i = m_new

		output[i:i + block_size] = O_i / l_i
	
	return output

