import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    clean_logits = X @ W_g
    noise_stddev = np.log1p(np.exp(X @ W_noise))
    logits = clean_logits + N * noise_stddev
    batch_size, num_experts = logits.shape
    k = min(k, num_experts)
    top_k_indices = np.argpartition(-logits, kth=k-1, axis=-1)[:, :k]
    mask = np.zeros_like(logits, dtype=bool)
    rows = np.arange(batch_size)[:, None]
    mask[rows, top_k_indices] = True
    masked_logits = np.where(mask, logits, -np.inf)
    max_vals = np.max(masked_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(masked_logits - max_vals)
    gates = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    gates = np.nan_to_num(gates)
    return gates.tolist()