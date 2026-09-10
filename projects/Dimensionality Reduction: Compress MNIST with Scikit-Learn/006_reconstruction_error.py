import numpy as np

def reconstruction_error(pca, X):
    X_rec = pca.inverse_transform(pca.transform(X))
    return float(np.mean((X - X_rec) ** 2))
