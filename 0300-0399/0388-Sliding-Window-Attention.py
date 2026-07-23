import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    seq_len, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)

    mask = np.ones((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) > window_size:
                mask[i, j] = 0

    scores = np.where(mask == 1, scores, -np.inf)

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = attn_weights @ V

    return np.round(output, 4)
