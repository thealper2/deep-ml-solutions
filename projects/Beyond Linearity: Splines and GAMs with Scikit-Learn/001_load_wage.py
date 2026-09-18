import os
import tempfile
import urllib.request
import pandas as pd

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"

def load_wage():
    path = os.path.join(tempfile.gettempdir(), 'Wage.csv')
    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)

def describe_wage(df):
    n = int(df.shape[0])
    columns = list(df.columns)
    age_range = (int(df['age'].min()), int(df['age'].max()))
    wage_mean = round(float(df['wage'].mean()), 2)
    n_education_levels = int(df['education'].nunique())

    return {
        'n': n,
        'columns': columns,
        'age_range': age_range,
        'wage_mean': wage_mean,
        'n_education_levels': n_education_levels,
    }