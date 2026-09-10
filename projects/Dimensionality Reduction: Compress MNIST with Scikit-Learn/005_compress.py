def compress(X, n_components, random_state=42):
    pca = fit_pca(X, n_components, random_state)
    return pca, pca.transform(X)
