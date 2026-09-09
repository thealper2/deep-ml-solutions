from sklearn.mixture import GaussianMixture

def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state
    ).fit(X)

def bic_curve(X, ks):
    return {k: float(fit_gmm(X, k).bic(X)) for k in ks}