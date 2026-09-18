import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.linear_model import LinearRegression


def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer([
        ('age', SplineTransformer(
            n_knots=age_knots, degree=3, knots='quantile', include_bias=False
        ), ['age']),
        ('year', SplineTransformer(
            n_knots=year_knots, degree=3, knots='uniform', include_bias=False
        ), ['year']),
        ('education', OneHotEncoder(drop='first'), ['education']),
    ])


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots=age_knots, year_knots=year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    return df[['age', 'year', 'education']], df['wage']


def gam_feature_count(model, X):
    pre = model.named_steps['columntransformer']
    out = pre.transform(X)
    return int(out.shape[1])