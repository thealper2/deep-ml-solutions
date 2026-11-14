def churn_rate_by(df, column):
    """Customers and churn rate for each value of column, highest rate first."""
    d = df[df[column].notna()]
    g = d.groupby(column).agg(customers=("churned", "size"), churn_rate=("churned", "mean")).reset_index()
    g = g.rename(columns={column: column})
    g = g.sort_values(["churn_rate", column], ascending=[False, True]).reset_index(drop=True)
    return g[[column, "customers", "churn_rate"]]