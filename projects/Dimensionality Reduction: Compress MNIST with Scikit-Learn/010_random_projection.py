import numpy as np
from sklearn.random_projection import GaussianRandomProjection

def random_projection(X, n_components, random_state=42, n_pairs=500):
    grp = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    Xr = grp.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), (n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    orig = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    proj = np.linalg.norm(Xr[pairs[:, 0]] - Xr[pairs[:, 1]], axis=1)
    ratios = proj / orig

    return {
        "X_reduced": Xr,
        "mean_ratio": float(np.mean(ratios)),
        "max_distortion": float(np.max(np.abs(ratios - 1))),
    }