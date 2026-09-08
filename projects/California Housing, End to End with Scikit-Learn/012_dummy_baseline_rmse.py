import numpy as np
from sklearn.dummy import DummyRegressor

def dummy_baseline_rmse(X, y):
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X, y)
    y_pred = dummy.predict(X)
    return rmse(y, y_pred)