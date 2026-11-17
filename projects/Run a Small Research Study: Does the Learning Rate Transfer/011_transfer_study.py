def transfer_study(rows, summary, small, big):
    """Per parametrization: the cost at width `big` of the learning rate tuned at width `small`."""
    result = {}
    for param in sorted(set(k[0] for k in summary.keys())):
        tuned = best_lr(summary[(param, small)])["grid_lr"]
        own = best_lr(summary[(param, big)])["grid_lr"]

        entry = {
            "tuned_log2": round(math.log2(tuned), 3) if tuned is not None else None,
            "big_best_log2": round(math.log2(own), 3) if own is not None else None,
        }

        # find the point for tuned at width big
        big_points = summary.get((param, big), [])
        tuned_point = None
        for p in big_points:
            if p["lr"] == tuned:
                tuned_point = p
                break

        if tuned_point is None or tuned_point["diverged"]:
            entry["diverged"] = True
            entry["penalty"] = None
        elif tuned == own:
            entry["diverged"] = False
            entry["penalty"] = {
                "diff": 0.0,
                "ci": (0.0, 0.0),
                "per_seed": [],
                "verdict": "same lr",
            }
        else:
            def per_example_for(lr):
                matching = [
                    r for r in rows
                    if r["param"] == param and r["width"] == big and r["lr"] == lr
                ]
                matching.sort(key=lambda r: r["seed"])
                return [r["per_example"] for r in matching]

            a = per_example_for(tuned)
            b = per_example_for(own)
            entry["diverged"] = False
            entry["penalty"] = paired_penalty(a, b)

        result[param] = entry
    return result