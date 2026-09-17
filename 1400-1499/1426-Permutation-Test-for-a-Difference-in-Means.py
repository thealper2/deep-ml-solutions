import numpy as np


def difference_in_means(x, y):
    return float(np.mean(x) - np.mean(y))


def permutation_null(x, y, n_perm, seed=0):
    x = np.asarray(x)
    y = np.asarray(y)
    pool = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = rng.permutation(pool)
        stats[i] = np.mean(shuffled[:n_x]) - np.mean(shuffled[n_x:])

    return stats


def permutation_pvalue(x, y, n_perm, seed=0):
    observed = difference_in_means(x, y)
    null = permutation_null(x, y, n_perm, seed)
    count = np.sum(np.abs(null) >= np.abs(observed))
    return round(float(count) / n_perm, 4)


def permutation_test(x, y, n_perm, alpha, seed=0):
    observed = difference_in_means(x, y)
    pvalue = permutation_pvalue(x, y, n_perm, seed)
    reject = pvalue < alpha
    return observed, pvalue, reject