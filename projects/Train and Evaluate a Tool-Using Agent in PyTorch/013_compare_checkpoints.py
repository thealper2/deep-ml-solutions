import math


def compare_checkpoints(old, new):
    """{'gap', 'se_paired', 'se_unpaired', 'verdict'} for two outcome matrices on the same tasks."""
    if len(old) != len(new):
        raise ValueError("different numbers of tasks")
    N = len(old)
    if N < 2:
        raise ValueError("need at least two tasks")

    old_means = np.array([np.mean(row) for row in old], dtype=float)
    new_means = np.array([np.mean(row) for row in new], dtype=float)
    d = new_means - old_means

    gap = float(d.mean())
    sd = float(np.std(d, ddof=1))
    se_paired = float(sd / np.sqrt(N))

    se_old = score_with_se(old)["se"]
    se_new = score_with_se(new)["se"]
    se_unpaired = float(np.sqrt(se_old ** 2 + se_new ** 2))

    if gap > 2 * se_paired:
        verdict = "new is better"
    elif gap < -2 * se_paired:
        verdict = "new is worse"
    else:
        verdict = "cannot tell"

    return {
        "gap": round(gap, 4),
        "se_paired": round(float(se_paired), 4),
        "se_unpaired": round(se_unpaired, 4),
        "verdict": verdict,
    }