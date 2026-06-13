import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    q = cache['q']
    k = cache['k']
    d_q = np.matmul(d_scores, k)
    d_k = np.matmul(d_scores.transpose(0, 2, 1), q)
    return {'d_q': d_q, 'd_k': d_k}
