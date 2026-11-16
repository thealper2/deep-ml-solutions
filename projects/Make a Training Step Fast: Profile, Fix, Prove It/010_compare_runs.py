def compare_runs(base_times, cand_times, warmup=2):
    """Speedup of the medians and whether the difference is beyond the interquartile ranges."""
    base = summarize_runs(base_times, warmup)
    cand = summarize_runs(cand_times, warmup)

    speedup = base["median"] / cand["median"]

    if cand["q3"] < base["q1"]:
        verdict = "faster"
    elif cand["q1"] > base["q3"]:
        verdict = "slower"
    else:
        verdict = "no measurable change"

    return {
        "speedup": speedup,
        "verdict": verdict,
        "base": base["median"],
        "cand": cand["median"],
    }