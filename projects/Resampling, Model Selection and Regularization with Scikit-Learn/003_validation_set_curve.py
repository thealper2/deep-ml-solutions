import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    poly = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression(),
    )
    return poly

def validation_set_curve(X, y, feature, degrees, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X[[feature]], y, test_size=0.5, random_state=random_state)
    curve = {}
    for degree in degrees:
        model = poly_model(degree)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        curve[degree] = round(float(mse), 1)

    return curve

def curve_spread(X, y, feature, degrees, seeds):
    spread = {}
    for d in degrees:
        vals = []
        for s in seeds:
            curve = validation_set_curve(X, y, feature, [d], random_state=s)
            vals.append(curve[d])
        spread[d] = round(max(vals) - min(vals), 1)

    return spread
