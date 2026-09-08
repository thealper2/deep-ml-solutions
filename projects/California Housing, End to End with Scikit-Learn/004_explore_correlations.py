def explore_correlations(df):
    corr_matrix = df.corr(numeric_only=True)
    correlations = corr_matrix["median_house_value"].drop("median_house_value").sort_values(ascending=False)
    return correlations