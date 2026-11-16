def canary_check(stable, canary, max_ratio=1.2, min_requests=200):
    """'wait', 'rollback' or 'promote' from the two groups' request and error counts."""
    if stable["requests"] < min_requests or canary["requests"] < min_requests:
        return "wait"

    stable_rate = stable["errors"] / stable["requests"]
    canary_rate = canary["errors"] / canary["requests"]

    if canary_rate > max_ratio * stable_rate:
        return "rollback"
    
    return "promote"