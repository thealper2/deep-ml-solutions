import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def pcr_model(n_components):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LinearRegression(),
    )


def explained_variance(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(Xs)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)


def pcr_curve(X, y, cv):
    p = X.shape[1]
    values = list(range(1, p + 1))
    means, ses = cv_curve(pcr_model, X, y, values, cv)
    return values, means, ses