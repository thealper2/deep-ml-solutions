def ci_gate(candidate, baseline, tolerance=0.015, min_count=100):
    """(passed, reasons) from overall and per-slice AUC rules."""
    reasons = []

    cand_overall = candidate["overall"]["auc"]
    base_overall = baseline["overall"]["auc"]
    if cand_overall is not None and base_overall is not None and cand_overall < base_overall:
        reasons.append("overall auc dropped")

    for name in sorted(baseline["slices"].keys()):
        base_slice = baseline["slices"][name]
        if name not in candidate["slices"]:
            reasons.append(f"slice missing: {name}")
            continue
        cand_slice = candidate["slices"][name]
        if cand_slice["n"] < min_count:
            continue
        if cand_slice["auc"] is None or base_slice["auc"] is None:
            continue
        if cand_slice["auc"] < base_slice["auc"] - tolerance:
            reasons.append(f"slice regressed: {name}")

    return (len(reasons) == 0, reasons)