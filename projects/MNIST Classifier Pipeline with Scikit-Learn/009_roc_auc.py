import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def roc_auc(y_true, scores):
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.argmax(tpr >= 0.9)
    return {
        "auc": auc,
        "fpr_at_recall_90": float(fpr[idx]),
    }