def paired_penalty(runs_a, runs_b, n_boot=2000, seed=0):
    """Mean loss difference A - B, a paired bootstrap interval over examples, per-seed differences and a verdict."""
    a = np.array(runs_a, dtype=float)
    b = np.array(runs_b, dtype=float)

    diff = a - b
    per_seed = diff.mean(axis=1)
    d = diff.mean(axis=0)
    overall = float(d.mean())

    n = d.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])

    if lo > 0 and np.all(per_seed > 0):
        verdict = "a is worse"
    elif hi < 0 and np.all(per_seed < 0):
        verdict = "a is better"
    else:
        verdict = "can't tell"

    return {
        "diff": round(overall, 4),
        "ci": (round(float(lo), 4), round(float(hi), 4)),
        "per_seed": [round(float(x), 4) for x in per_seed],
        "verdict": verdict,
    }