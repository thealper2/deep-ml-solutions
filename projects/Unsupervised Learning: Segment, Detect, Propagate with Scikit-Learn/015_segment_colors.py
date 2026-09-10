import numpy as np

def segment_colors(image, k=4, random_state=42):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    km = fit_kmeans(pixels, k, random_state)
    segmented = km.cluster_centers_[km.labels_].reshape(h, w, 3)
    return segmented