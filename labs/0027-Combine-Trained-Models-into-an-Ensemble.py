import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_ensemble(base_models, X_val, y_val):
    """
    Combine K pre-trained base classifiers into an ensemble that predicts class
    labels on new data.

    The harness has already trained the base models on a separate training set.
    You DO NOT retrain them. Instead, use their predictions on (X_val, y_val) to
    figure out how to combine them, then return a `predict` callable that the
    harness will apply to held-out test data.

    Args:
        base_models: list of K trained sklearn-style classifiers. Each has:
            - .predict(X) -> np.ndarray of integer class labels, shape (n,)
            - .predict_proba(X) -> np.ndarray of shape (n, n_classes)
        X_val: numpy array of shape (n_val, n_features) -- standardized
        y_val: numpy array of shape (n_val,) -- integer class labels

    Returns:
        predict: callable taking X (n, n_features) and returning a numpy array
                 of integer class labels, shape (n,)
    """
    val_probas = []
    for model in base_models:
        val_probas.append(model.predict_proba(X_val))

    stacked_probas = np.hstack(val_probas)

    meta_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=0.1,
        random_state=42,
    )
    meta_model.fit(stacked_probas, X_val)

    def predict(X):
        test_probas = []
        for model in base_model:
            test_probas.append(model.predict_proba(X))

        stacked_test = np.hstack(test_proba)
        return meta_model.predict(stacked_test)

    return predict
