import numpy as np

def ovr_predict(scores):
    """
    scores: (n_samples, n_classes) decision scores, one column per one-vs-rest classifier.
    Returns: (n_samples,) int array of predicted classes.
    """
    scores = np.asarray(scores)
    return np.argmax(scores, axis=1)

def ovo_predict(pair_scores, n_classes):
    """
    pair_scores: (n_samples, n_pairs) decision scores, one column per pair (i, j), i < j,
                 in lexicographic order. Positive votes for j, otherwise for i.
    Returns: (n_samples,) int array of predicted classes (ties -> smallest index).
    """
    pair_scores = np.asarray(pair_scores)
    n_samples = pair_scores.shape[0]

    n_pairs = n_classes * (n_classes - 1) // 2

    votes = np.zeros((n_samples, n_classes), dtype=int)

    pair_idx = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            mask = pair_scores[:, pair_idx] > 0
            votes[mask, j] += 1
            votes[~mask, i] += 1
            pair_idx += 1

    return np.argmax(votes, axis=1)