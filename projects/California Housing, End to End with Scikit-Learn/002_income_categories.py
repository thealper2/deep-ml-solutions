import numpy as np
import pandas as pd

def income_categories(df):
    bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(df['median_income'], bins=bins, labels=labels, right=True).astype(int)