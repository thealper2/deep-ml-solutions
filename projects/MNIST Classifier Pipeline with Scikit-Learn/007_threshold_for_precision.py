import numpy as np
from sklearn.metrics import precision_recall_curve

def threshold_for_precision(y_true, scores, target=0.90):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    idx = np.argmax(precisions >= target)
    return float(thresholds[idx])