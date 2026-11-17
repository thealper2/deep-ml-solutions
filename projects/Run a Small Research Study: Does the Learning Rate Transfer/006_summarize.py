def summarize(rows):
    """{(param, width): [one point per lr, sorted by lr]} with mean, std, n and diverged."""
    groups = {}
    for r in rows:
        key = (r["param"], r["width"])
        groups.setdefault(key, {}).setdefault(r["lr"], []).append(r)

    out = {}
    for key, by_lr in groups.items():
        points = []
        for lr in sorted(by_lr.keys()):
            runs = by_lr[lr]
            n = len(runs)
            diverged = any(r["diverged"] for r in runs)
            if diverged:
                points.append({
                    "lr": lr,
                    "mean": None,
                    "std": None,
                    "n": n,
                    "diverged": True,
                })
            else:
                losses = np.array([r["val_loss"] for r in runs], dtype=float)
                mean = round(float(losses.mean()), 4)
                std = round(float(losses.std(ddof=1)), 4) if n >= 2 else None
                points.append({
                    "lr": lr,
                    "mean": mean,
                    "std": std,
                    "n": n,
                    "diverged": False,
                })
        out[key] = points
    return out