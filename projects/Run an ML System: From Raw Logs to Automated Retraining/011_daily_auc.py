import numpy as np
from sklearn.metrics import roc_auc_score


def daily_auc(events, probs, labels, min_labels=100):
    """{day: AUC} on matured labels for days with enough labels and both classes."""
    groups = {}
    for ev, p in zip(events, probs):
        eid = ev["event_id"]
        if eid not in labels:
            continue
        day = ev["ts"] // 1440
        groups.setdefault(day, []).append((float(p), int(labels[eid])))

    result = {}
    for day in sorted(groups.keys()):
        items = groups[day]
        if len(items) < min_labels:
            continue
        y = [label for _, label in items]
        if len(set(y)) < 2:
            continue
        p = [prob for prob, _ in items]
        result[day] = round(float(roc_auc_score(y, p)), 4)
    
    return result