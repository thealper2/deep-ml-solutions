import pandas as pd


def fix_types(df):
    """Drop exact duplicates and give every column a usable type."""
    out = df.drop_duplicates().reset_index(drop=True).copy()

    for col in ("plan", "contract", "payment_method"):
        out[col] = out[col].str.strip().str.lower()

    out["monthly_charge"] = (
        out["monthly_charge"].str.replace("$", "", regex=False).astype(float)
    )

    out["signup_date"] = pd.to_datetime(out["signup_date"])

    out["churned"] = out["churned"].map({"Yes": 1, "No": 0}).astype(int)

    return out