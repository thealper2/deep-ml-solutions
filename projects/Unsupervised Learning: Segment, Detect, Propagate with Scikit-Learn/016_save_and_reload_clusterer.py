import joblib

def save_and_reload_clusterer(kmeans, rep_labels, path):
    joblib.dump({"kmeans": kmeans, "rep_labels": rep_labels}, path)
    return joblib.load(path)