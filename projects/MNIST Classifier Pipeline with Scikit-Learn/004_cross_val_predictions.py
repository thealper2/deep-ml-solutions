from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    return cross_val_predict(clone(clf), X, y, cv=cv, method=method)