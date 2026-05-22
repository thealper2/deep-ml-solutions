import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    X = np.array(X)
    W_query = np.array(W_query)
    W_key = np.array(W_key)
    W_value = np.array(W_value)

    B, T, d_in = X.shape
    d_out = W_query.shape[1]

    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_out)

    mask = np.tril(np.ones((T, T)))
    scores = np.where(mask == 0, -np.inf, scores)

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return (probs @ X).tolist()