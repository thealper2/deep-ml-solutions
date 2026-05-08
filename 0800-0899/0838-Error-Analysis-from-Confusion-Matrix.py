import numpy as np

def error_analysis(cm):
    """
    Analyze errors in a multi-class confusion matrix.

    Args:
        cm: n x n confusion matrix (list of lists or numpy array),
            cm[i][j] = count of true class i predicted as j.

    Returns:
        Dict with keys 'per_class_error_rate', 'overall_error_rate',
        'most_confused_pair', 'worst_class'.
    """
    cm = np.array(cm)
    n = cm.shape[0]
    
    row_sums = np.sum(cm, axis=1)
    diag = np.diag(cm)
    
    per_class_error_rate = []
    for i in range(n):
        if row_sums[i] == 0:
            per_class_error_rate.append(0.0)
        else:
            per_class_error_rate.append((row_sums[i] - diag[i]) / row_sums[i])
    
    total = np.sum(cm)
    if total == 0:
        overall_error_rate = 0.0
    else:
        overall_error_rate = 1 - (np.sum(diag) / total)
    
    max_confusion = -1
    most_confused_pair = [0, 1]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > max_confusion:
                max_confusion = cm[i, j]
                most_confused_pair = [i, j]
            elif i != j and cm[i, j] == max_confusion and max_confusion != -1:
                if i < most_confused_pair[0] or (i == most_confused_pair[0] and j < most_confused_pair[1]):
                    most_confused_pair = [i, j]
    
    worst_class = np.argmax(per_class_error_rate)
    
    return {
        'per_class_error_rate': [float(i) for i in per_class_error_rate],
        'overall_error_rate': float(overall_error_rate),
        'most_confused_pair': most_confused_pair,
        'worst_class': worst_class,
    }