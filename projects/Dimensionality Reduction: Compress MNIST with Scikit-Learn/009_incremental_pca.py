import numpy as np
from sklearn.decomposition import IncrementalPCA

def incremental_pca(X, n_components, n_batches=10):
    ipca = IncrementalPCA(n_components=n_components)
    for batch in np.array_split(X, n_batches):
        ipca.partial_fit(batch)

    return ipca