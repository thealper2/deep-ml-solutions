from sklearn.cluster import KMeans

def fit_kmeans(X, k, random_state=42):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    kmeans.fit(X)
    return kmeans