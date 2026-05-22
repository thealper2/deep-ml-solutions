import numpy as np

def sample_name(P, itos, seed, max_len=100):
    rng = np.random.default_rng(seed)
    P = np.array(P)

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    prob_matrix = P / row_sums

    zero_rows = (P.sum(axis=1) == 0)
    prob_matrix[zero_rows] = 0
    prob_matrix[zero_rows, 0] = 1

    cdf_matrix = np.cumsum(prob_matrix, axis=1)

    result = []
    ix = 0

    for _ in range(max_len):
        r = rng.random()
        next_ix = np.searchsorted(cdf_matrix[ix], r)

        if next_ix == 0:
            break
        
        result.append(itos[next_ix])
        ix = next_ix

    return ''.join(result)