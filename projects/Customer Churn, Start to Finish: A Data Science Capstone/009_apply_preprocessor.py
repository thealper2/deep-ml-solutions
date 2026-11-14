import numpy as np


def apply_preprocessor(prep, df):
    """Fill, standardize and one-hot encode df with the numbers in prep; return (X, names)."""
    cols = []
    names = []

    for col in prep["numeric"]:
        filled = df[col].fillna(prep["median"][col])
        vals = (filled - prep["mean"][col]) / prep["std"][col]
        cols.append(vals.to_numpy(dtype=float))
        names.append(col)

    for col in prep["categorical"]:
        for level in prep["levels"][col]:
            vals = (df[col] == level).to_numpy(dtype=float)
            cols.append(vals)
            names.append(f"{col}={level}")

    X = np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=float)
    return X, names