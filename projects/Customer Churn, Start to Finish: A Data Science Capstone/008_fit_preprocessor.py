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