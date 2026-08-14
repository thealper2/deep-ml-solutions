import numpy as np

def nn_1nn_forward(X_train, y_train, X_test):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    predictions = []
    
    for x in X_test:
        distances = np.sum((X_train - x) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        predictions.append(int(y_train[nearest_idx]))
    
    return predictions
