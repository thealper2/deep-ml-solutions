def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'accuracy': metrics['accuracy'],
    }
