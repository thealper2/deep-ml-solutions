import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = X.shape

    w = np.full(n_samples, 1 / n_samples)

    clfs = []
    for _ in range(n_clf):
        clf = {}
        min_error = float('inf')

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                for polarity in (1, -1):
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    error = np.sum(w[y != predictions])

                    if error < min_error:
                        min_error = error
                        clf['polarity'] = polarity
                        clf['threshold'] = threshold
                        clf['feature_index'] = feature_i

        EPS = 1e-10
        clf['alpha'] = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

        predictions = np.ones(n_samples)
        fv = X[:, clf['feature_index']]
        if clf['polarity'] == 1:
            predictions[fv < clf['threshold']] = -1
        else:
            predictions[fv >= clf['threshold']] = -1

        w *= np.exp(-clf['alpha'] * y * predictions)
        w /= np.sum(w)

        clfs.append(clf)

    return clfs
