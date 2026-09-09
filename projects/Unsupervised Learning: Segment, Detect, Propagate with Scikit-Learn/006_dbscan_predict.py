from sklearn.neighbors import KNeighborsClassifier

def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_indices = dbscan.core_sample_indices_
    core_samples = dbscan.components_
    core_labels = dbscan.labels_[core_indices]
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(core_samples, core_labels)
    
    return knn.predict(X_new)