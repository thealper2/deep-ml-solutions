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