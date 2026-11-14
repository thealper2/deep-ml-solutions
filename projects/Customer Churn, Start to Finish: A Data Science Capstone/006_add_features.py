import pandas as pd


def add_features(df, drop_columns, snapshot="2025-01-01"):
    """Add tenure_months and tickets_per_month, then drop drop_columns."""
    out = df.copy()
    snap = pd.Timestamp(snapshot)
    days = (snap - out["signup_date"]).dt.days
    out["tenure_months"] = (days // 30).astype(int)
    denom = out["tenure_months"].where(out["tenure_months"] != 0, 1)
    out["tickets_per_month"] = out["support_tickets"] / denom
    out = out.drop(columns=list(drop_columns))
    return out