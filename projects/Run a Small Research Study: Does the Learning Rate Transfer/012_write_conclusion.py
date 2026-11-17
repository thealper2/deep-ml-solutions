NAMES = {"sp": "Standard parametrization", "mup": "muP"}


def write_conclusion(study, slopes, small, big):
    """The claims the evidence supports, one string per line."""
    lines = []
    transfers = {}

    for param in sorted(study.keys()):
        name = NAMES[param]
        entry = study[param]

        if entry["diverged"]:
            lines.append(
                f"{name}: the learning rate tuned at width {small} diverges at width {big}."
            )
            transfers[param] = False
            continue

        penalty = entry["penalty"]
        verdict = penalty["verdict"]

        if verdict == "same lr":
            lines.append(
                f"{name}: width {big} has the same best learning rate as width {small}."
            )
            transfers[param] = True
        elif verdict == "a is worse":
            lo, hi = penalty["ci"]
            k = len(penalty["per_seed"])
            lines.append(
                f"{name}: the learning rate tuned at width {small} costs "
                f"{penalty['diff']:+.4f} validation loss at width {big} "
                f"(95% CI {lo:.4f} to {hi:.4f}, worse on all {k} seeds)."
            )
            transfers[param] = False
        else:
            lo, hi = penalty["ci"]
            lines.append(
                f"{name}: no measurable cost from the learning rate tuned at width {small} "
                f"({penalty['diff']:+.4f}, 95% CI {lo:.4f} to {hi:.4f})."
            )
            transfers[param] = True

    for param in sorted(slopes.keys()):
        name = NAMES[param]
        slope = slopes[param]["slope"] if isinstance(slopes[param], dict) else slopes[param]
        lines.append(
            f"{name}: best log2 learning rate changes by {slope:+.2f} per doubling of width."
        )

    mup_ok = transfers.get("mup", False)
    sp_ok = transfers.get("sp", False)

    if mup_ok and not sp_ok:
        lines.append(
            "Conclusion: on this task the learning rate transfers from small to large width "
            "under muP and not under the standard parametrization."
        )
    elif mup_ok and sp_ok:
        lines.append(
            "Conclusion: the learning rate transfers under both parametrizations here, "
            "so this task does not separate them."
        )
    else:
        lines.append(
            "Conclusion: inconclusive; muP did not transfer the learning rate on this task."
        )

    return lines