def lr_sweep(params, widths, lrs, seeds, data, steps=300, batch_size=128):
    """One row per (param, width, lr, seed) run, in that loop order."""
    rows = []
    for param in params:
        for width in widths:
            for lr in lrs:
                for seed in seeds:
                    r = train_run(width, param, lr, seed, data, steps=steps, batch_size=batch_size)
                    rows.append({
                        "param": param,
                        "width": width,
                        "lr": lr,
                        "seed": seed,
                        "val_loss": r["val_loss"],
                        "per_example": r["per_example"],
                        "diverged": r["diverged"],
                    })
    return rows