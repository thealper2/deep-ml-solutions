import numpy as np


def cross_validation_split(data: np.ndarray, k: int) -> list:
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(data)
        test_indices = data[start:end]
        train_indices = np.concatenate([data[:start], data[end:]])
        folds.append([train_indices.tolist(), test_indices.tolist()])
    return folds


data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
k = 5

folds = cross_validation_split(data, k)
print(folds)
