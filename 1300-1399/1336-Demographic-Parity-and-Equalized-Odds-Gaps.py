import numpy as np

def fairness_gaps(y_true, y_pred, group):
    """Audit demographic parity and equalized odds across groups."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    groups = np.unique(group)

    positive_rates = []
    tprs = []
    fprs = []

    for g in groups:
        mask = group == g
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        pr = np.mean(y_pred_g)
        positive_rates.append(pr)

        pos_mask = y_true_g == 1
        if np.sum(pos_mask) > 0:
            tpr = np.mean(y_pred_g[pos_mask])
        else:
            tpr = 0.0

        tprs.append(tpr)

        neg_mask = y_true_g == 0
        if np.sum(neg_mask) > 0:
            fpr = np.mean(y_pred_g[neg_mask])
        else:
            fpr = 0.0

        fprs.append(fpr)

    dp_gap = np.max(positive_rates) - np.min(positive_rates)
    tpr_gap = np.max(tprs) - np.min(tprs)
    fpr_gap = np.max(fprs) - np.min(fprs)
    eo_gap = max(tpr_gap, fpr_gap)

    return {
        'groups': [int(g) for g in groups],
        'positive_rate': [round(float(x), 4) for x in positive_rates],
        'tpr': [round(float(x), 4) for x in tprs],
        'fpr': [round(float(x), 4) for x in fprs],
        'dp_gap': round(float(dp_gap), 4),
        'tpr_gap': round(float(tpr_gap), 4),
        'fpr_gap': round(float(fpr_gap), 4),
        'eo_gap': round(float(eo_gap), 4)
    }