def speedup_report(steps, min_share=0.01):
    """Attribute the end-to-end speedup to each fix in the order they were applied."""
    baseline = steps[0][1]
    final = steps[-1][1]
    total_saved = baseline - final
    speedup = baseline / final

    fixes = []
    for i in range(1, len(steps)):
        label, after = steps[i]
        before = steps[i - 1][1]
        saved = before - after
        share = saved / total_saved if total_saved != 0 else 0.0
        fspeedup = before / after
        fixes.append({
            "fix": label,
            "saved_us": saved,
            "share": share,
            "speedup": fspeedup,
        })

    biggest = None
    if fixes:
        best = fixes[0]
        for f in fixes[1:]:
            if f["saved_us"] > best["saved_us"]:
                best = f
        biggest = best["fix"]

    did_not_matter = []
    for i in range(1, len(steps)):
        label, after = steps[i]
        before = steps[i - 1][1]
        saved = before - after
        if saved < min_share * before:
            did_not_matter.append(label)

    return {
        "speedup": speedup,
        "fixes": fixes,
        "biggest": biggest,
        "did_not_matter": did_not_matter,
    }