import numpy as np

def score_lr_model(model, X, y):
    y_pred = predict_lr_model(model, X)
    return evaluate_regression(y, y_pred)
