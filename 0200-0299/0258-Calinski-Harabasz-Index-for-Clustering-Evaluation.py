import numpy as np

def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Calinski-Harabasz Index for clustering evaluation.
    
    Args:
        X: numpy array of shape (n_samples, n_features) containing data points
        labels: numpy array of shape (n_samples,) containing cluster assignments
    
    Returns:
        float: Calinski-Harabasz score (higher is better)
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0

    global_centroid = np.mean(X, axis=0)

    between_dispersion = 0.0

    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_size = len(cluster_points)
        centroid = np.mean(cluster_points, axis=0)
        diff = centroid - global_centroid
        between_dispersion += cluster_size * np.dot(diff, diff)

    within_dispersion = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        diff = cluster_points - centroid
        within_dispersion += np.sum(diff ** 2)

    if within_dispersion == 0:
        return 0.0
    
    score = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
    return float(score)
