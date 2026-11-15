import numpy as np


def shadow_compare(prod_probs, cand_probs, labels, min_labeled=200, min_gain=0.01, threshold=0.5):
    """Agreement, accuracies on labeled requests, and promote / hold / reject."""
    prod_probs = np.asarray(prod_probs, dtype=float)
    cand_probs = np.asarray(cand_probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    prod_pred = (prod_probs >= threshold).astype(int)
    cand_pred = (cand_probs >= threshold).astype(int)

    agreement = float((prod_pred == cand_pred).mean()) if len(prod_pred) > 0 else 0.0

    known = ~np.isnan(labels)
    labeled = int(known.sum())

    if labeled < min_labeled:
        return {
            "agreement": agreement,
            "labeled": labeled,
            "prod_accuracy": None,
            "cand_accuracy": None,
            "decision": "hold",
        }

    y = labels[known].astype(int)
    prod_acc = float((prod_pred[known] == y).mean())
    cand_acc = float((cand_pred[known] == y).mean())

    if cand_acc - prod_acc >= min_gain:
        decision = "promote"
    else:
        decision = "reject"

    return {
        "agreement": agreement,
        "labeled": labeled,
        "prod_accuracy": prod_acc,
        "cand_accuracy": cand_acc,
        "decision": decision,
    }