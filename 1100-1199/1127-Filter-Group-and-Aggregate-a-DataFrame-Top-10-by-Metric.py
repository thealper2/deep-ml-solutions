import pandas as pd

def solution(df):
    completed = df[df['status'] == 'completed']
    result = completed.groupby('region', as_index=False)['amount'].sum()
    result = result.sort_values('amount', ascending=False).head(10)
    result = result.reset_index(drop=True)
    return result
