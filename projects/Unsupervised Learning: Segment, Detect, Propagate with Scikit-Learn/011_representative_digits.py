import numpy as np

def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state)
    distances = kmeans.transform(X_train)
    rep_idx = np.argmin(distances, axis=0)
    return kmeans, rep_idx