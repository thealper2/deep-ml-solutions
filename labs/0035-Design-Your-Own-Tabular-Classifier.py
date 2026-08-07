import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, X_val, y_val):
    """
    Design your own tabular binary classifier on real census data.

    X_* are pandas DataFrames with MIXED dtypes (numeric + categorical).
    Features are RAW — you own imputation, encoding, scaling, imbalance
    handling, model choice, and calibration.

    Target: predict whether income is >50K (class 1, minority).

    Args:
        X_train: pd.DataFrame (n_train, n_features)
        y_train: np.ndarray (n_train,) binary labels {0, 1}
        X_val:   pd.DataFrame (n_val, n_features)
        y_val:   np.ndarray (n_val,) binary labels {0, 1}

    Returns:
        predict_proba: callable predict_proba(X) -> np.ndarray (n,)
            Positive-class scores for a DataFrame X with the same
            columns as X_train. Finite scores; ranking is what matters
            for PR-AUC.
    """
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val])
    
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_combined, y_combined)
    
    X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val)
    
    cal_classifier = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    cal_classifier.fit(X_val_transformed, y_val)
    
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    
    base_probs = pipeline.named_steps['classifier'].predict_proba(X_val_transformed)[:, 1]
    base_probs = base_probs.reshape(-1, 1)
    
    calibrator.fit(base_probs, y_val)
    
    def predict_proba(X):
        X_transformed = pipeline.named_steps['preprocessor'].transform(X)
        base_probs = pipeline.named_steps['classifier'].predict_proba(X_transformed)[:, 1]
        base_probs = base_probs.reshape(-1, 1)
        cal_probs = calibrator.predict_proba(base_probs)[:, 1]
        return cal_probs
    
    return predict_proba
