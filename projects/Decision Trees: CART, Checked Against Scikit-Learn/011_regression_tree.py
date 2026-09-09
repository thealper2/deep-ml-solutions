import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def regression_tree(X, y, max_depth=2, random_state=42):
    reg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    reg.fit(X, y)
    pred = reg.predict(X)
    n_distinct = len(np.unique(pred))
    mse = mean_squared_error(y, pred)

    return {
        "model": reg,
        "n_distinct_predictions": n_distinct,
        "train_mse": float(mse),
    }