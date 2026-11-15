import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)

def slice_report(y_true, p, slices):
    """Overall and per-slice counts and AUC (rounded to 4, None for a one-class group)."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    slices = np.asarray(slices)

    overall = {
        "n": int(len(y_true)),
        "auc": _auc(y_true, p),
    }

    result = {}
    for name in sorted(set(slices.tolist())):
        mask = slices == name
        result[name] = {
            "n": int(mask.sum()),
            "auc": _auc(y_true[mask], p[mask]),
        }

    return {"overall": overall, "slices": result}
