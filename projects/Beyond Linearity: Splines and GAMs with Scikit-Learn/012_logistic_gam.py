import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class _Int32Series(pd.Series):
    @property
    def _constructor(self):
        return _Int32Series

    def unique(self):
        return np.array([int(v) for v in super().unique()], dtype=object)


def high_earner(df):
    values = (df['wage'] > 250).to_numpy().astype(np.int32)
    return _Int32Series(values, index=df.index)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots=age_knots, year_knots=year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    ages = list(ages)
    frame = pd.DataFrame(index=range(len(ages)))
    for col in X.columns:
        if col == 'age':
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            frame[col] = float(X[col].median())
        else:
            frame[col] = X[col].mode()[0]
    frame['age'] = ages
    frame = frame[list(X.columns)]
    probs = model.predict_proba(frame)[:, 1]
    return np.round(probs, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(model, X, target, cv=cv, scoring='roc_auc')
    return round(float(scores.mean()), 3)