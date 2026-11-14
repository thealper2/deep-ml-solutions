# ---- provided: the raw export (do not edit)
import numpy as np
import pandas as pd


def load_raw_customers(n=2000, seed=7):
    """A subscription company's customer export on 2025-01-01, as messy as real ones are."""
    rs = np.random.RandomState(seed)
    tenure = rs.randint(1, 61, size=n)
    signup = pd.Timestamp("2025-01-01") - pd.to_timedelta(tenure * 30 + rs.randint(0, 30, size=n), unit="D")
    plan = rs.choice(["basic", "standard", "premium"], size=n, p=[0.5, 0.3, 0.2])
    contract = rs.choice(["monthly", "annual"], size=n, p=[0.65, 0.35])
    base = np.where(plan == "basic", 20.0, np.where(plan == "standard", 45.0, 80.0))
    charge = np.round(base + rs.normal(0, 4, size=n), 2)
    age = rs.randint(18, 75, size=n).astype(float)
    tickets = rs.poisson(1.2, size=n).astype(int)
    logins = rs.poisson(12, size=n).astype(float)
    payment = rs.choice(["card", "bank", "paypal"], size=n, p=[0.6, 0.25, 0.15])
    score = (-0.9 + 1.5 * (contract == "monthly") - 0.04 * tenure + 0.55 * tickets
             - 0.12 * logins + 0.5 * (plan == "basic"))
    churned = rs.rand(n) < 1.0 / (1.0 + np.exp(-score))
    survey = np.where(churned & (rs.rand(n) < 0.8), rs.randint(1, 6, size=n), np.nan)

    def messy(values, share):
        styles = [str.upper, str.title, lambda s: s + " ", lambda s: " " + s]
        pick = rs.randint(0, len(styles), size=n)
        change = rs.rand(n) < share
        return [styles[pick[i]](v) if change[i] else v for i, v in enumerate(values)]

    charge_text = [("$%.2f" % c) if dollar else str(c) for c, dollar in zip(charge, rs.rand(n) < 0.7)]
    df = pd.DataFrame({
        "customer_id": np.arange(10001, 10001 + n),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "plan": messy(plan, 0.25),
        "contract": messy(contract, 0.2),
        "monthly_charge": charge_text,
        "age": age,
        "support_tickets": tickets,
        "logins_last_30d": logins,
        "payment_method": payment,
        "cancel_survey_score": survey,
        "churned": np.where(churned, "Yes", "No"),
    })
    df.loc[rs.rand(n) < 0.03, "monthly_charge"] = None
    df.loc[rs.rand(n) < 0.02, "plan"] = None
    df.loc[rs.rand(n) < 0.04, "age"] = np.nan
    df.loc[rs.rand(n) < 0.01, "age"] = rs.choice([0.0, 199.0])
    df.loc[rs.rand(n) < 0.05, "logins_last_30d"] = np.nan
    df.loc[rs.rand(n) < 0.01, "support_tickets"] = -1
    df = pd.concat([df, df.iloc[rs.choice(n, size=40, replace=False)]], ignore_index=True)
    return df.iloc[rs.permutation(len(df))].reset_index(drop=True)
# ---- end provided


def audit_table(df):
    """Report rows, columns, duplicate rows, missing counts and non-numeric columns."""
    missing = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            missing[col] = cnt

    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": missing,
        "non_numeric": non_numeric,
    }
