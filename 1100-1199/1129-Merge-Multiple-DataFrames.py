import pandas as pd

def solution(df1, df2, df3):
    merged = pd.merge(df1, df2, on='emp_id', how='inner')
    result = pd.merge(merged, df3, on='emp_id', how='left')
    result = result[['emp_id', 'name', 'dept', 'salary']]
    result = result.reset_index(drop=True)
    return result
