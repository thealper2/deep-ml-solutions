import numpy as np

def engram_context_gating(h: np.ndarray, e: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Implement Engram context-aware gating mechanism.
    
    Args:
        h: Hidden states of shape (T, d)
        e: Retrieved memory embeddings of shape (T, d_mem)
        W_K: Key projection matrix of shape (d_mem, d)
        W_V: Value projection matrix of shape (d_mem, d)
        eps: Small constant for numerical stability in RMSNorm
    
    Returns:
        Gated output of shape (T, d)
    """
    k = e @ W_K
    v = e @ W_V
    h_norm = h / np.sqrt(np.mean(h ** 2, axis=1, keepdims=True) + eps)
    k_norm = k / np.sqrt(np.mean(k ** 2, axis=1, keepdims=True) + eps)
    scale = 1 / np.sqrt(h.shape[1])
    scaled_dot = np.sum(h_norm * k_norm, axis=1, keepdims=True) * scale
    gate = 1 / (1 + np.exp(-scaled_dot))
    output = gate * v
    return output