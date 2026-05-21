import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    T = attn_weights.shape[0]
    attn_weights = np.array(attn_weights)

    if method == 'tril':
        mask = np.tril(np.ones((T, T)))
        masked = attn_weights * mask
        row_sums = masked.sum(axis=1, keepdims=True)
        result = masked / row_sums
        return result.tolist()

    elif method == 'triu':
        eps = 1e-10
        logits = np.log(attn_weights + eps)
        mask = np.triu(np.ones((T, T)), k=1)
        logits = np.where(mask == 1, -np.inf, logits)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        result = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return result.tolist()
