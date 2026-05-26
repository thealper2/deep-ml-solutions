import numpy as np

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    seq_len, d_model = X.shape
    num_heads = len(W_queries)
    d_k = W_queries[0].shape[1]
    d_v = W_value.shape[1]

    K = X @ W_key
    V = X @ W_value

    head_outputs = []

    for head_idx in range(num_heads):
        Q = X @ W_queries[head_idx]
        scores = Q @ K.T / np.sqrt(d_k)
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        context = attn_weights @ V
        head_outputs.append(context)

    concat_out = np.concatenate(head_outputs, axis=1)
    output = concat_out @ W_out
    return np.round(output, 4)