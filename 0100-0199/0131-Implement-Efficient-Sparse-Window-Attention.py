import numpy as np


def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    seq_len = Q.shape[0]
    d_k = Q.shape[1]
    d_v = V.shape[1]

    if scale_factor is None:
        scale_factor = np.sqrt(d_k)

    output = np.zeros((seq_len, d_v))

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        q_i = Q[i:i + 1, :]
        K_window = K[start:end, :]
        V_window = V[start:end, :]
        scores = np.dot(q_i, K_window.T) / scale_factor
        scores_stable = scores - np.max(scores)
        exp_scores = np.exp(scores_stable)
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        output[i] = np.dot(attention_weights, V_window)[0]

    return output