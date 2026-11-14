"""
Customer Churn, Start to Finish: A Data Science Capstone — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  audit_table ──
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

# ── Step 002  fix_types ──
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

# ── Step 003  mark_invalid ──
def mark_invalid(df):
    """Turn ages outside 18-100 and negative ticket counts into missing values."""
    out = df.copy()
    age_ok = out["age"].between(18, 100)
    out.loc[~age_ok & out["age"].notna(), "age"] = float("nan")
    out.loc[out["support_tickets"] < 0, "support_tickets"] = float("nan")
    return out

# ── Step 004  churn_rate_by ──
def churn_rate_by(df, column):
    """Customers and churn rate for each value of column, highest rate first."""
    d = df[df[column].notna()]
    g = d.groupby(column).agg(customers=("churned", "size"), churn_rate=("churned", "mean")).reset_index()
    g = g.rename(columns={column: column})
    g = g.sort_values(["churn_rate", column], ascending=[False, True]).reset_index(drop=True)
    return g[[column, "customers", "churn_rate"]]

# ── Step 005  find_leaky_columns ──
def find_leaky_columns(df, target="churned", min_present=20):
    """Columns filled in only when the target has one particular value."""
    leaky = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if not s.isna().any():
            continue
        present = s.notna().sum()
        if present < min_present:
            continue
        targets = df.loc[s.notna(), target].dropna().unique()
        if len(targets) == 1:
            leaky.append(col)
    return leaky

# ── Step 006  add_features ──
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

# ── Step 007  split_customers ──
from sklearn.model_selection import train_test_split


def split_customers(df, test_size=0.25, seed=0):
    """Stratified random split on churned; returns (train, test) with fresh indexes."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["churned"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)

# ── Step 008  fit_preprocessor ──
def fit_preprocessor(train, numeric, categorical):
    """Medians, means, standard deviations and category levels learned from train."""
    median = {}
    mean = {}
    std = {}
    levels = {}

    for col in numeric:
        s = train[col]
        med = float(s.median())
        median[col] = med
        filled = s.fillna(med)
        mean[col] = float(filled.mean())
        sd = float(filled.std(ddof=0))
        std[col] = sd if sd != 0 else 1.0

    for col in categorical:
        vals = sorted(train[col].dropna().unique().tolist())
        levels[col] = vals

    return {
        "numeric": list(numeric),
        "categorical": list(categorical),
        "median": median,
        "mean": mean,
        "std": std,
        "levels": levels,
    }

# ── Step 009  apply_preprocessor ──
import numpy as np


def apply_preprocessor(prep, df):
    """Fill, standardize and one-hot encode df with the numbers in prep; return (X, names)."""
    cols = []
    names = []

    for col in prep["numeric"]:
        filled = df[col].fillna(prep["median"][col])
        vals = (filled - prep["mean"][col]) / prep["std"][col]
        cols.append(vals.to_numpy(dtype=float))
        names.append(col)

    for col in prep["categorical"]:
        for level in prep["levels"][col]:
            vals = (df[col] == level).to_numpy(dtype=float)
            cols.append(vals)
            names.append(f"{col}={level}")

    X = np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=float)
    return X, names

# ── Step 010  baseline_metrics ──
import numpy as np


def baseline_metrics(y_train, y_test):
    """Accuracy and recall of always predicting the majority class of y_train."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    ones = int((y_train == 1).sum())
    zeros = int((y_train == 0).sum())
    majority = 1 if ones > zeros else 0

    preds = np.full(len(y_test), majority)
    accuracy = float((preds == y_test).mean())

    actual_pos = int((y_test == 1).sum())
    if actual_pos == 0:
        recall = 0.0
    else:
        recall = float(((preds == 1) & (y_test == 1)).sum() / actual_pos)

    base_rate = float(ones / len(y_train))

    return {
        "majority_class": int(majority),
        "accuracy": accuracy,
        "recall": recall,
        "base_rate": base_rate,
    }

# ── Step 011  train_model ──
from sklearn.linear_model import LogisticRegression


def train_model(X, y, C=1.0):
    """Fit and return LogisticRegression(C=C, max_iter=2000, tol=1e-8)."""
    model = LogisticRegression(C=C, max_iter=2000, tol=1e-8)
    model.fit(X, y)
    return model

# ── Step 012  evaluate_model ──
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_model(y_true, p, threshold=0.5):
    """AUC, precision, recall, F1 and the four counts at the threshold."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    auc = float(roc_auc_score(y_true, p))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

# ── Step 013  choose_threshold ──
import numpy as np


def choose_threshold(y_true, p, cost_fp, cost_fn):
    """(threshold, cost) with the lowest fp * cost_fp + fn * cost_fn over 0.05..0.95."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    best_t = None
    best_cost = None
    for i in range(1, 20):
        t = round(i * 0.05, 2)
        pred = (p >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp * cost_fp + fn * cost_fn
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_t = t

    return (best_t, best_cost)

# ── Step 014  targeting_report ──
import numpy as np
import math

def targeting_report(y_true, p, top_frac=0.2):
    """Contact the top_frac riskiest customers; report reach, recall, precision and lift."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y_true)
    k = max(1, math.ceil(top_frac * n))

    order = np.argsort(-p, kind="stable")
    chosen = order[:k]
    reached = int((y_true[chosen] == 1).sum())
    total_pos = int((y_true == 1).sum())

    recall = reached / total_pos if total_pos > 0 else 0.0
    precision = reached / k
    overall = total_pos / n
    lift = precision / overall if overall > 0 else 0.0

    return {
        "contacted": int(k),
        "churners_reached": reached,
        "recall": float(recall),
        "precision": float(precision),
        "lift": float(lift),
    }

# ── Step 015  top_drivers ──
def top_drivers(model, names, k=5):
    """The k largest coefficients by size: (name, rounded coefficient, 'raises' or 'lowers')."""
    coefs = model.coef_[0]
    pairs = list(zip(names, coefs))
    pairs.sort(key=lambda x: (-abs(x[1]), x[0]))
    out = []
    for name, c in pairs[:k]:
        out.append((name, round(float(c), 3), "raises" if c > 0 else "lowers"))
    return out

# ── Scaffold (runner) ──
"""Customer churn, start to finish.

Story: take a messy customer export, clean it, find what goes with churn, remove a column that
gives the answer away, split honestly, beat a baseline with a model, then turn the model's scores
into a decision the business can act on.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def main():
    # ---- 1. Get to know the data and clean it ----
    raw = load_raw_customers()
    audit = audit_table(raw)
    print("raw table:", audit["rows"], "rows,", audit["columns"], "columns,", audit["duplicate_rows"], "duplicate rows")
    print("missing values:", audit["missing"])
    clean = mark_invalid(fix_types(raw))
    print("after cleaning:", len(clean), "customers, churn rate", round(float(clean["churned"].mean()), 3))

    # ---- 2. Explore and prepare ----
    for column in ["contract", "plan"]:
        table = churn_rate_by(clean, column)
        print("churn by", column + ":", {str(k): round(float(v), 3) for k, v in zip(table[column], table["churn_rate"])})
    leaky = find_leaky_columns(clean)
    print("columns that give the answer away:", leaky)
    data = add_features(clean, drop_columns=["customer_id", "signup_date"] + leaky)
    train, test = split_customers(data, test_size=0.25, seed=0)
    print("train:", len(train), "test:", len(test))

    # ---- 3. Model and evaluate honestly ----
    numeric = ["monthly_charge", "age", "support_tickets", "logins_last_30d", "tenure_months", "tickets_per_month"]
    categorical = ["plan", "contract", "payment_method"]
    prep = fit_preprocessor(train, numeric, categorical)
    X_train, names = apply_preprocessor(prep, train)
    X_test, _ = apply_preprocessor(prep, test)
    y_train, y_test = train["churned"].to_numpy(), test["churned"].to_numpy()
    base = baseline_metrics(y_train, y_test)
    print("baseline: accuracy", round(base["accuracy"], 3), "recall", round(base["recall"], 3))
    model = train_model(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    scores = evaluate_model(y_test, p_test)
    print("model at 0.5: auc", round(scores["auc"], 3), "precision", round(scores["precision"], 3),
          "recall", round(scores["recall"], 3))

    # ---- 4. Turn it into a decision ----
    # Choose the threshold on predictions for training rows that each model never saw
    # (5-fold out-of-fold), so the test set is still used only once, here below.
    p_oof = cross_val_predict(LogisticRegression(max_iter=2000, tol=1e-8), X_train, y_train,
                              cv=5, method="predict_proba")[:, 1]
    threshold, _ = choose_threshold(y_train, p_oof, cost_fp=20, cost_fn=60)
    at_best = evaluate_model(y_test, p_test, threshold)
    test_cost = at_best["fp"] * 20 + at_best["fn"] * 60
    print("threshold chosen on training rows:", threshold, "| on the test set: cost", test_cost,
          "recall", round(at_best["recall"], 3), "precision", round(at_best["precision"], 3))
    report = targeting_report(y_test, p_test, top_frac=0.2)
    print("contact the riskiest 20%:", report["contacted"], "customers reach",
          report["churners_reached"], "churners, lift", round(report["lift"], 2))
    print("what moves churn most:")
    for name, coef, direction in top_drivers(model, names, k=5):
        print("  ", name, coef, direction)


if __name__ == "__main__":
    main()
