def release_check(base_times, cand_times, base_loss, cand_loss, loss_tol=0.01, min_speedup=1.05, warmup=2):
    """Ship only a candidate that is faster beyond the noise, by enough, without changing the loss."""
    cmp = compare_runs(base_times, cand_times, warmup)
    reasons = []

    if cmp["verdict"] != "faster":
        reasons.append("not faster beyond the noise: " + cmp["verdict"])

    if cmp["speedup"] < min_speedup:
        reasons.append(f"speedup {cmp['speedup']:.3f} is below {min_speedup}")

    if abs(cand_loss - base_loss) > loss_tol:
        reasons.append(f"loss changed by {cand_loss - base_loss:+.4f}")

    return {
        "ship": len(reasons) == 0,
        "speedup": round(cmp["speedup"], 4),
        "reasons": reasons,
    }