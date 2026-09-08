"""
California Housing, End to End with Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_housing ──
import os
import tempfile
import urllib.request
import tarfile
import pandas as pd

def load_housing():
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    temp_dir = tempfile.gettempdir()
    tgz_path = os.path.join(temp_dir, "housing.tgz")

    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(url, tgz_path)

    with tarfile.open(tgz_path) as tar:
        csv_file = tar.extractfile("housing/housing.csv")
        if csv_file is not None:
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError("housing/housing.csv not found in the tarball")

    return df

# ── Step 002  income_categories ──
import numpy as np
import pandas as pd

def income_categories(df):
    bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(df['median_income'], bins=bins, labels=labels, right=True).astype(int)

# ── Step 003  stratified_split ──
from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state=42):
    strata = income_categories(df)
    train_set, test_set = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=strata
    )
    return train_set, test_set

# ── Step 004  explore_correlations ──
def explore_correlations(df):
    corr_matrix = df.corr(numeric_only=True)
    correlations = corr_matrix["median_house_value"].drop("median_house_value").sort_values(ascending=False)
    return correlations

# ── Step 005  add_ratio_features ──
def add_ratio_features(df):
    df_copy = df.copy()
    df_copy['rooms_per_house'] = df_copy['total_rooms'] / df_copy['households']
    df_copy['bedrooms_ratio'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
    df_copy['people_per_house'] = df_copy['population'] / df_copy['households']
    return df_copy

# ── Step 006  split_features_labels ──
def split_features_labels(df):
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    return X, y

# ── Step 007  ClusterSimilarity ──
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# ── Step 008  numeric_pipeline ──
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# ── Step 009  categorical_pipeline ──
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

# ── Step 010  build_preprocessing ──
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

# ── Step 011  rmse ──
import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ── Step 012  dummy_baseline_rmse ──
import numpy as np
from sklearn.dummy import DummyRegressor

def dummy_baseline_rmse(X, y):
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X, y)
    y_pred = dummy.predict(X)
    return rmse(y, y_pred)

# ── Step 013  cross_val_rmse ──
import numpy as np
from sklearn.model_selection import cross_val_score

def cross_val_rmse(model, X, y, cv=3):
    scores = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return {
        "scores": scores.tolist(),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

# ── Step 014  linear_model ──
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def linear_model(preprocessing):
    return make_pipeline(preprocessing, LinearRegression())

# ── Step 015  forest_model ──
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

def forest_model(preprocessing, n_estimators=50, random_state=42):
    return make_pipeline(preprocessing, RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))

# ── Step 016  random_search ──
from sklearn.model_selection import RandomizedSearchCV

def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    param_distribs = {
        "columntransformer__geo__n_clusters": list(range(3, 11)),
        "randomforestregressor__max_features": list(range(2, 9)),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search

# ── Step 017  test_rmse ──
def test_rmse(model, test_set):
    X_test, y_test = split_features_labels(add_ratio_features(test_set))
    y_pred = model.predict(X_test)
    return rmse(y_test, y_pred)

# ── Step 018  bootstrap_rmse_ci ──
import numpy as np

def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    boot_rmse = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rmse.append(rmse(y_true[idx], y_pred[idx]))

    low = float(np.percentile(boot_rmse, 100 * alpha / 2))
    high = float(np.percentile(boot_rmse, 100 * (1 - alpha / 2)))
    return low, high

# ── Step 019  feature_importances ──
def feature_importances(search, k=5):
    best_estimator = search.best_estimator_
    preprocessor = best_estimator.steps[0][1]
    feature_names = preprocessor.get_feature_names_out()
    importances = best_estimator.steps[-1][1].feature_importances_
    pairs = sorted(zip(importances, feature_names), reverse=True)[:k]
    return [(float(round(imp, 3)), name) for imp, name in pairs]

# ── Step 020  worst_errors ──
def worst_errors(model, df, k=5):
    X_test, y_test = split_features_labels(add_ratio_features(df))
    y_pred = model.predict(X_test)
    
    errors = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'abs_error': np.abs(y_test - y_pred)
    }, index=df.index)
    
    return errors.nlargest(k, 'abs_error')

# ── Step 021  save_and_reload ──
import joblib

def save_and_reload(model, path):
    joblib.dump(model, path)
    return joblib.load(path)

# ── Step 022  predict_new ──
import numpy as np
import pandas as pd

def predict_new(model, districts):
    df = pd.DataFrame(districts)
    df = add_ratio_features(df)
    predictions = model.predict(df)
    return [float(np.round(pred)) for pred in predictions]

# ── Scaffold (runner) ──
"""California Housing, end to end with scikit-learn (Hands-On ML, chapter 2).

Story: get the data, split it honestly, look at it, build a preprocessing pipeline
that cannot leak, beat a dummy baseline with a linear model and then a random
forest under cross-validation, tune the whole pipeline, report the test RMSE with
a bootstrap interval, inspect what the model relies on and where it fails, then
save it, reload it and predict on raw new districts. A 4,000-district subsample
keeps the run under the time budget; the numbers track the book's.
"""
import os
import tarfile
import tempfile
import urllib.request
import numpy as np
import pandas as pd


def main() -> None:
    housing = load_housing()
    print(f"loaded {len(housing):,} districts, {housing.shape[1]} columns; "
          f"missing total_bedrooms: {int(housing['total_bedrooms'].isna().sum())}")

    # ---- 1. Split first, then look ----
    train_set, test_set = stratified_split(housing)
    print(f"train {len(train_set):,} / test {len(test_set):,} (stratified on income category)")
    corr = explore_correlations(add_ratio_features(train_set))
    print("top correlations with value:", ", ".join(f"{k} {v:+.2f}" for k, v in corr.head(3).items()))

    sample = train_set.sample(4000, random_state=42)
    X, y = split_features_labels(add_ratio_features(sample))

    # ---- 2. Baseline, then models under cross-validation ----
    dummy = dummy_baseline_rmse(X, y)
    print(f"\ndummy (predict the mean)   RMSE {dummy:>10,.0f}")
    lin = cross_val_rmse(linear_model(build_preprocessing()), X, y)
    print(f"linear regression   CV   RMSE {lin['mean']:>10,.0f}  (+/- {lin['std']:,.0f})")
    forest = forest_model(build_preprocessing(), n_estimators=30)
    fr = cross_val_rmse(forest, X, y)
    train_r = rmse(y, forest.fit(X, y).predict(X))
    print(f"random forest       CV   RMSE {fr['mean']:>10,.0f}  (+/- {fr['std']:,.0f}); "
          f"on its own training data {train_r:,.0f} -> it overfits, trust the CV number")

    # ---- 3. Tune the whole pipeline ----
    search = random_search(forest_model(build_preprocessing(), n_estimators=30), X, y, n_iter=3)
    print(f"\nrandom search best CV RMSE {-search.best_score_:,.0f} with {search.best_params_}")

    # ---- 4. The test set, once ----
    final_model = search.best_estimator_
    X_test, y_test = split_features_labels(add_ratio_features(test_set))
    pred = final_model.predict(X_test)
    low, high = bootstrap_rmse_ci(y_test, pred, n_boot=200)
    print(f"TEST RMSE {test_rmse(final_model, test_set):,.0f}   95% bootstrap CI [{low:,.0f}, {high:,.0f}]")
    print("what it relies on:", ", ".join(f"{n} {i:.3f}" for i, n in feature_importances(search, k=4)))
    worst = worst_errors(final_model, test_set, k=3)
    print("worst misses (actual / predicted):",
          ", ".join(f"{a:,.0f} / {p:,.0f}" for a, p in zip(worst["actual"], worst["predicted"])))

    # ---- 5. Ship ----
    path = os.path.join(tempfile.gettempdir(), "california_housing_model.pkl")
    served = save_and_reload(final_model, path)
    districts = [
        {"longitude": -122.2, "latitude": 37.8, "housing_median_age": 30.0, "total_rooms": 2000.0,
         "total_bedrooms": 400.0, "population": 1000.0, "households": 380.0, "median_income": 5.5,
         "ocean_proximity": "NEAR BAY"},
        {"longitude": -119.5, "latitude": 36.5, "housing_median_age": 20.0, "total_rooms": 1500.0,
         "total_bedrooms": None, "population": 900.0, "households": 300.0, "median_income": 2.1,
         "ocean_proximity": "INLAND"},
        {"longitude": -118.4, "latitude": 34.0, "housing_median_age": 40.0, "total_rooms": 2500.0,
         "total_bedrooms": 450.0, "population": 1100.0, "households": 420.0, "median_income": 9.0,
         "ocean_proximity": "<1H OCEAN"},
    ]
    preds = predict_new(served, districts)
    for d, p in zip(districts, preds):
        print(f"  {d['ocean_proximity']:<10} income {d['median_income']:>4}  ->  ${p:,.0f}")
    print(f"\nsaved to {os.path.basename(path)}; reloaded model reproduces the test score: "
          f"{abs(test_rmse(served, test_set) - test_rmse(final_model, test_set)) < 1e-6}")


if __name__ == "__main__":
    main()
