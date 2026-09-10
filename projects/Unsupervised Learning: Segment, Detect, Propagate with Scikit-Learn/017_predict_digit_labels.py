import numpy as np

def predict_digit_labels(bundle, images):
    X = np.asarray(images, dtype=float).reshape(len(images), -1)
    clusters = bundle["kmeans"].predict(X)
    return [int(v) for v in bundle["rep_labels"][clusters]]