import numpy as np
from sklearn.decomposition import PCA

def randomized_pca(X, n_components, random_state=42):
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(X)
    exact, _ = compress(X, n_components)
    gap = abs(float(pca.explained_variance_ratio_.sum()) - float(exact.explained_variance_ratio_.sum()))
    return {"pca": pca, "variance_gap": gap}