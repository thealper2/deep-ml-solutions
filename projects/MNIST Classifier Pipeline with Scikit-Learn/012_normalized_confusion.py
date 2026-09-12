import numpy as np
from sklearn.metrics import confusion_matrix

def normalized_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)), normalize="true")
    return np.nan_to_num(cm)