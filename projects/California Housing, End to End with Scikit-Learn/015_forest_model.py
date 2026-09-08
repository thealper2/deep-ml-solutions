from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

def forest_model(preprocessing, n_estimators=50, random_state=42):
    return make_pipeline(preprocessing, RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
