import numpy as np

def qk_norm_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.
    
    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        temperature: Temperature scaling parameter (default: 1.0)
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    eps = 1e-8
    q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, eps)
    Q_norm = Q / q_norms

    k_norms = np.linalg.norm(K, axis=1, keepdims=True)
    k_norms = np.maximum(k_norms, eps)
    K_norm = K / k_norms

    scores = (Q_norm @ K_norm.T) / temperature

    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    output = attention_weights @ V
    return output, attention_weights