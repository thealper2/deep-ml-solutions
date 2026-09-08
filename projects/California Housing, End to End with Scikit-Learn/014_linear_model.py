from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def linear_model(preprocessing):
    return make_pipeline(preprocessing, LinearRegression())

