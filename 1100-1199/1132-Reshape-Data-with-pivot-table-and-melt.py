import pandas as pd

def solution(df):
    df_agg = df.groupby(['date', 'product'], as_index=False)['sales'].sum()
    wide = df_agg.pivot(index='date', columns='product', values='sales').fillna(0)
    l = wide.reset_index().melt(id_vars='date', var_name='product', value_name='sales')
    result = l.sort_values(['date', 'product']).reset_index(drop=True)
    return result
