import numpy as np

def evaluate_at_threshold(y_true, scores, threshold):
    pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(pred.sum()),
    }