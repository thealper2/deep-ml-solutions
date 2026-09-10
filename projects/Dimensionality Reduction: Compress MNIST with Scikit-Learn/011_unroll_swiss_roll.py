import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(n_samples, noise=0.2, random_state=random_state)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=random_state)
    X_2d = lle.fit_transform(X)

    c0 = abs(float(np.corrcoef(t, X_2d[:, 0])[0, 1]))
    c1 = abs(float(np.corrcoef(t, X_2d[:, 1])[0, 1]))

    return {
        "X_2d": X_2d,
        "t_correlation": max(c0, c1),
    }