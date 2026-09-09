from sklearn.cluster import DBSCAN

def fit_dbscan(X, eps=0.2, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)

def dbscan_summary(dbscan):
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for l in labels if l == -1)
    n_core = len(dbscan.core_sample_indices_)
    return {
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "n_core": int(n_core)
    }