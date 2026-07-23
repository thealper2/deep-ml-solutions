import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    q = x_new @ W_Q
    k = x_new @ W_K
    v = x_new @ W_V

    if cache is None:
        K_cache = k.reshape(1, -1)
        V_cache = v.reshape(1, -1)
    else:
        K_cache, V_cache = cache
        K_cache = np.vstack([K_cache, k])
        V_cache = np.vstack([V_cache, v])

    d_k = W_Q.shape[1]
    scores = q @ K_cache.T / np.sqrt(d_k)

    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    attn_weights = exp_scores / np.sum(exp_scores)

    output = attn_weights @ V_cache
    return output, (K_cache, V_cache)
