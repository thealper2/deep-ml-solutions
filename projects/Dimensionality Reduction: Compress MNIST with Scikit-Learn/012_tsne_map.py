import numpy as np
from sklearn.manifold import TSNE

def tsne_map(X, y, n=1000, random_state=42):
    X_sub = X[:n]
    y_sub = y[:n]

    tsne = TSNE(n_components=2, init="pca", random_state=random_state)
    X_2d = tsne.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = np.array([X_2d[y_sub == c].mean(axis=0) for c in classes])

    within = np.mean([
        np.mean(np.linalg.norm(X_2d[y_sub == c] - centroids[i], axis=1))
        for i, c in enumerate(classes)
    ])

    between = np.mean([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(classes))
        for j in range(i + 1, len(classes))
    ])

    return {
        "X_2d": X_2d,
        "separation": float(between / within),
    }