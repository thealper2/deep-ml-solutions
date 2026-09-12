from sklearn.metrics import precision_score, recall_score, f1_score

def precision_recall_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(precision), float(recall), float(f1)