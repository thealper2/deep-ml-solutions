import pandas as pd

def solution(df):
    df = df.copy()

    col_missing_frac = df.isnull().mean()
    cols_to_drop = col_missing_frac[col_missing_frac > 0.5].index
    df = df.drop(columns=cols_to_drop)

    row_missing_frac = df.isnull().mean(axis=1)
    rows_to_drop = row_missing_frac[row_missing_frac > 0.5].index
    df = df.drop(index=rows_to_drop)

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.reset_index(drop=True)
    return df
