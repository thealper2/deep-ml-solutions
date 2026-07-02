import numpy as np

def predict_from_scores(scores):
    return np.where(scores > 0, 1, -1)
