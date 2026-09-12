from sklearn.model_selection import cross_val_score

def multiclass_cv_accuracy(model, X, y, cv=3):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())