import numpy as np
import pandas as pd


def _base_frame(X, column, n_rows):
    frame = pd.DataFrame(index=range(n_rows))
    for col in X.columns:
        if col == column:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            frame[col] = float(X[col].median())
        else:
            frame[col] = X[col].mode()[0]
    return frame


def partial_effect(model, X, column, grid_values):
    grid_values = list(grid_values)
    frame = _base_frame(X, column, len(grid_values))
    frame[column] = grid_values
    frame = frame[list(X.columns)]
    preds = np.asarray(model.predict(frame)).ravel()
    centered = preds - preds.mean()
    return np.round(centered, 2)


def education_effect(model, X):
    levels = sorted(X['education'].unique())
    grid_values = list(levels)
    effect = partial_effect(model, X, 'education', grid_values)
    return {str(level): float(v) for level, v in zip(levels, effect)}


def gam_summary(model, X):
    age_effect = partial_effect(model, X, 'age', age_grid()['age'])
    age_range = float(np.max(age_effect) - np.min(age_effect))

    years = list(range(2003, 2010))
    year_effect = partial_effect(model, X, 'year', years)
    year_range = float(np.max(year_effect) - np.min(year_effect))

    levels = sorted(X['education'].unique())
    edu_effect = partial_effect(model, X, 'education', list(levels))
    edu_range = float(np.max(edu_effect) - np.min(edu_effect))

    return {
        'age_range': round(age_range, 1),
        'year_range': round(year_range, 1),
        'education_range': round(edu_range, 1),
    }