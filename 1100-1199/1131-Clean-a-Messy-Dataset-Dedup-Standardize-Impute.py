import pandas as pd

def solution(df):
    df['name'] = df['name'].apply(lambda x: x.strip().capitalize())
    df['date'] = df['date'].str.strip()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop=True)
    df['value'] = df['value'].fillna(df['value'].mean())

    return df
