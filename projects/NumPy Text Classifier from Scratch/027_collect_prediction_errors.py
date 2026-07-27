def collect_prediction_errors(texts: list, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    false_positives = []
    false_negatives = []
    for text, y_t, y_p in zip(texts, y_true, y_pred):
        if y_t == 1 and y_p == 0:
            false_negatives.append(text)
        elif y_t == 0 and y_p == 1:
            false_positives.append(text)

    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }
