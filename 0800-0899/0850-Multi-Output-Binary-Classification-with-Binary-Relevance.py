import numpy as np

def multi_output_classify(X_train, Y_train, X_test, lr=0.1, n_iter=1000):
    """
    Train one logistic regression per output column (binary relevance) and
    return predictions on X_test as a nested list of ints.
    """
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    n_samples, n_features = X_train.shape
    n_outputs = Y_train.shape[1]

    X_train_bias = np.hstack([np.ones((n_samples, 1)), X_train])
    X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    n_features_with_bias = n_features + 1

    predictions = []

    for j in range(n_outputs):
        weights = np.zeros(n_features_with_bias)
        
        for _ in range(n_iter):
            logits = X_train_bias @ weights
            probs = 1 / (1 + np.exp(-logits))
            gradient = (1 / n_samples) * (X_train_bias.T @ (probs - Y_train[:, j]))
            weights -= lr * gradient

        logits_test = X_test_bias @ weights
        probs_test = 1 / (1 + np.exp(-logits_test))
        pred_j = (probs_test >= 0.5).astype(int)
        predictions.append(pred_j)

    return np.array(predictions).T.tolist()