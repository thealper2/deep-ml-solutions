def histogram_quantile(x, n_bins, lo, hi, q_frac):
    """Quantile from pooled bin counts; error <= (hi - lo) / n_bins.

    Return the right edge of the first bin whose cumulative count reaches
    q_frac * len(x).
    """
    counts, bin_edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    cumsum = np.cumsum(counts)
    target = q_frac * len(x)
    idx = np.searchsorted(cumsum, target)
    if idx >= n_bins:
        return float(bin_edges[-1])

    return float(bin_edges[idx + 1])
