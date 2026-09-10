import numpy as np

def explained_variance_curve(pca, ks):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return {k: round(float(cumsum[k - 1]), 4) for k in ks}