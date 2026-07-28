def quantile_balance_update(s, bias, k):
    """QB (Eq. 14): bhat_j = -(the (q+1)-th largest of s[:, j] - cutoffs),
    q = m*k // n; return bhat - mean(bhat).
    """
    m, n = s.shape
    q = (m * k) // n
    routes, cutoffs = topk_cutoffs(s, bias, k)
    margins = s - cutoffs[:, None]
    bhat = np.zeros(n)
    for j in range(n):
        col_margins = margins[:, j]
        sorted_margins = np.sort(col_margins)[::-1]
        bhat[j] = -sorted_margins[q]

    bhat = bhat - np.mean(bhat)
    return bhat
