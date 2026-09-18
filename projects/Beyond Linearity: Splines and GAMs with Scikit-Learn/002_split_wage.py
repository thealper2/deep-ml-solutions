import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_wage(df, test_size=0.25, random_state=0):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def age_xy(df):
    return df[['age']], df['wage']

def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({'age': np.linspace(lo, hi, n)})