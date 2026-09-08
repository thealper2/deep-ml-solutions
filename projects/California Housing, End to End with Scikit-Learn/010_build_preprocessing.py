from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )

    cat_pipeline = categorical_pipeline()
    num_pipeline = numeric_pipeline()

    preprocessor = ColumnTransformer([
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", ClusterSimilarity(n_clusters=n_clusters, gamma=gamma, random_state=random_state), ["latitude", "longitude"]),
        ("cat", cat_pipeline, ["ocean_proximity"])
    ], remainder=num_pipeline)

    return preprocessor