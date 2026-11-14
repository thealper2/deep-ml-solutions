def mark_invalid(df):
    """Turn ages outside 18-100 and negative ticket counts into missing values."""
    out = df.copy()
    age_ok = out["age"].between(18, 100)
    out.loc[~age_ok & out["age"].notna(), "age"] = float("nan")
    out.loc[out["support_tickets"] < 0, "support_tickets"] = float("nan")
    return out