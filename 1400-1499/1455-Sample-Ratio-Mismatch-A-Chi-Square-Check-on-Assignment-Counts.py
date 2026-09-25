import numpy as np
from math import exp, sqrt, pi, erfc


def chi2_sf(x, df):
    """Survival function P(X > x) of a chi-square distribution with integer df (provided)."""
    if x <= 0:
        return 1.0
    if df % 2 == 0:
        term, total = 1.0, 1.0
        for i in range(1, df // 2):
            term *= (x / 2.0) / i
            total += term
        return exp(-x / 2.0) * total
    total = erfc(sqrt(x / 2.0))
    term = sqrt(2.0 * x / pi) * exp(-x / 2.0)
    for i in range(1, (df - 1) // 2 + 1):
        if i > 1:
            term *= x / (2.0 * i - 1.0)
        total += term
    return total


def expected_counts(counts, ratios):
    total = float(sum(counts))
    s = float(sum(ratios))
    return [total * r / s for r in ratios]

def srm_chi_square(counts, ratios):
    expected = expected_counts(counts, ratios)
    chi2 = sum((o - e) ** 2 / e for o, e in zip(counts, expected))
    k = len(counts)
    df = k - 1
    p = chi2_sf(chi2, df)
    return round(chi2, 4), round(p, 8)

def srm_check(counts, ratios, alpha=0.001):
    chi2, p = srm_chi_square(counts, ratios)
    return chi2, p, p < alpha