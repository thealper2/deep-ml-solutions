import numpy as np
from sklearn.model_selection import cross_val_score

def cross_val_rmse(model, X, y, cv=3):
    scores = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return {
        "scores": scores.tolist(),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
