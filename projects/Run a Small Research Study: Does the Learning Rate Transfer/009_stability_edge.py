def stability_edge(points, best_log2):
    """The largest learning rate below the first divergence, and its distance from the best one."""
    max_stable_log2 = None
    for p in points:
        if p["diverged"]:
            break
        max_stable_log2 = round(math.log2(p["lr"]), 3)

    if max_stable_log2 is None:
        return {"max_stable_log2": None, "headroom": None}

    headroom = None
    if best_log2 is not None:
        headroom = round(max_stable_log2 - best_log2, 3)

    return {"max_stable_log2": max_stable_log2, "headroom": headroom}