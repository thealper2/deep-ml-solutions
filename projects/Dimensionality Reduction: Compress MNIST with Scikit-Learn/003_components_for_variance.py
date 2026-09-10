import numpy as np

def components_for_variance(pca, threshold=0.95):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumsum >= threshold)) + 1
