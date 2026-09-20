import numpy as np

def qda_predict(X_train, y_train, X_test):
    """
    Train a QDA classifier on (X_train, y_train) and predict labels for X_test.
    Returns a list of predicted class labels (Python ints).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.unique(y_train)
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]

    scores = np.zeros((n_test, len(classes)))

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        prior_c = X_c.shape[0] / n_train
        mean_c = np.mean(X_c, axis=0)

        cov_c = np.cov(X_c, rowvar=False) + 1e-6 * np.eye(n_features)
        inv_cov_c = np.linalg.inv(cov_c)
        sign, logdet_c = np.linalg.slogdet(cov_c)

        X_centered = X_test - mean_c

        mahalanobis = -0.5 * np.sum(np.dot(X_centered, inv_cov_c) * X_centered, axis=1)

        scores[:, idx] = mahalanobis - 0.5 * logdet_c + np.log(prior_c)

    return classes[np.argmax(scores, axis=1)]        