import numpy as np
import math

def targeting_report(y_true, p, top_frac=0.2):
    """Contact the top_frac riskiest customers; report reach, recall, precision and lift."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y_true)
    k = max(1, math.ceil(top_frac * n))

    order = np.argsort(-p, kind="stable")
    chosen = order[:k]
    reached = int((y_true[chosen] == 1).sum())
    total_pos = int((y_true == 1).sum())

    recall = reached / total_pos if total_pos > 0 else 0.0
    precision = reached / k
    overall = total_pos / n
    lift = precision / overall if overall > 0 else 0.0

    return {
        "contacted": int(k),
        "churners_reached": reached,
        "recall": float(recall),
        "precision": float(precision),
        "lift": float(lift),
    }