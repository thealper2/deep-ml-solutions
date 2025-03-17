import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)


def log_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def gradient_descent(X, y, weights, learning_rate=0.1, num_iterations=100):
    m = len(y)
    weights = np.array(weights, dtype="float")
    for _ in range(num_iterations):
        y_pred = predict(X, weights)
        gradient = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradient

    return weights


def classify(X, weights, threshold=0.5):
    y_pred = predict(X, weights)
    return (y_pred >= threshold).astype(int)


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_b = np.c_[np.ones((X.shape[0], bias)), X]
    learning_rate = 0.1
    num_iterations = 100
    w = gradient_descent(X_b, y, weights, learning_rate, num_iterations)
    predictions = classify(X_b, w)
    return predictions


X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
weights = np.array([1, 1])
bias = 0

result = predict_logistic(X, weights, bias)
print(result)
