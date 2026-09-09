from sklearn.metrics import silhouette_score

def silhouette_curve(X, ks):
    return {k: float(silhouette_score(X, fit_kmeans(X, k).labels_)) for k in ks}

def best_k_by_silhouette(curve):
    return max(curve.keys(), key=lambda k: (curve[k], -k))