import numpy as np

def predict_labels(x, params):
    scores = compute_scores(x, params)
    preds = predict_from_scores(scores)
    return preds
