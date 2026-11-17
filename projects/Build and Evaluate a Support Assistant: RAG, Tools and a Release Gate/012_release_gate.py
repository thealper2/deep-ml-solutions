def release_gate(old, new, golden, tolerance=0.0):
    """Ship only with no unsafe actions, no category regressions and no failed critical cases."""
    reasons = []

    if new["unsafe_actions"] > 0:
        reasons.append(f"unsafe actions: {new['unsafe_actions']}")

    old_cats = old.get("by_category", {})
    new_cats = new.get("by_category", {})
    for cat in sorted(old_cats.keys()):
        if cat in new_cats:
            if new_cats[cat] < old_cats[cat] - tolerance:
                reasons.append(f"category regressed: {cat}")

    old_cases = old.get("cases", {})
    new_cases = new.get("cases", {})
    for case in golden:
        if case.get("critical") and old_cases.get(case["id"]) and not new_cases.get(case["id"]):
            reasons.append(f"critical case failed: {case['id']}")

    return {"ship": len(reasons) == 0, "reasons": reasons}
