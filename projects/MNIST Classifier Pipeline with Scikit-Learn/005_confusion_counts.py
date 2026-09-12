from sklearn.metrics import confusion_matrix

def confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    TN, FP, FN, TP = cm.ravel()
    return {"TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP)}