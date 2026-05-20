import numpy as np

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    if not words:
        return 0.0

    total_neg_log_lik = 0.0
    total_bigrams = 0

    for word in words:
        padded = '.' + word + '.'
        for i in range(len(padded) - 1):
            idx1 = stoi[padded[i]]
            idx2 = stoi[padded[i + 1]]
            prob = P[idx1, idx2]
            total_neg_log_lik += -np.log(prob)
            total_bigrams += 1

    if total_bigrams == 0:
        return 0.0

    return round(total_neg_log_lik / total_bigrams, 4)