from sklearn.decomposition import PCA

def fit_pca(X, n_components=None, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    return pca