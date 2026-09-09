def inertia_curve(X, ks):
    return {k: float(fit_kmeans(X, k).inertia_) for k in ks}