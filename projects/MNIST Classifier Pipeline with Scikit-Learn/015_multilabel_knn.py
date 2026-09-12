import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def multilabel_knn(X, Y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, Y)
    return knn

def multilabel_f1(Y_true, Y_pred):
    return float(f1_score(Y_true, Y_pred, average="macro"))