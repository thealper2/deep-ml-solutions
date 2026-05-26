import numpy as np

def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
	"""
	Detect attention sink tokens from multi-head attention weight matrices.
	
	Args:
		attn_weights: Attention weights of shape (num_heads, seq_len, seq_len)
		threshold: Minimum average received attention to qualify as a sink
		
	Returns:
		Dictionary with 'sink_positions', 'avg_attention_received', and 'sink_scores'
	"""
	num_heads, seq_len, _ = attn_weights.shape
	avg_attention_received = np.mean(attn_weights, axis=(0, 1))
	sink_positions = np.where(avg_attention_received >= threshold)[0].tolist()
	sink_scores = avg_attention_received[sink_positions]
	return {
		'sink_positions': sink_positions,
		'avg_attention_received': np.round(avg_attention_received, 4).tolist(),
		'sink_scores': np.round(sink_scores, 4).tolist(),
	}