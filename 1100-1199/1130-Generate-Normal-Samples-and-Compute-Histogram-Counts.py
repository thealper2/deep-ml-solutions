import numpy as np

def fn(seed, mean, std, n, bins):
    np.random.seed(seed)
    samples = np.random.normal(loc=mean, scale=std, size=n)
    counts, bin_edges = np.histogram(samples, bins=bins)
    return counts.tolist(), bin_edges.astype(float).tolist()
