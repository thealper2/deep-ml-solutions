def top_drivers(model, names, k=5):
    """The k largest coefficients by size: (name, rounded coefficient, 'raises' or 'lowers')."""
    coefs = model.coef_[0]
    pairs = list(zip(names, coefs))
    pairs.sort(key=lambda x: (-abs(x[1]), x[0]))
    out = []
    for name, c in pairs[:k]:
        out.append((name, round(float(c), 3), "raises" if c > 0 else "lowers"))
    return out