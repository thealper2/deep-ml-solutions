from math import comb


def pass_k(outcomes, k):
    """{'pass_at_k': at least one of k trials succeeds, 'pass_hat_k': all k succeed}, averaged over tasks."""
    if not outcomes:
        raise ValueError("outcomes is empty")
    n = len(outcomes[0])
    for row in outcomes:
        if len(row) != n:
            raise ValueError("rows differ in length")
    if not (1 <= k <= n):
        raise ValueError("k out of range")

    denom = comb(n, k)
    at_sum = 0.0
    hat_sum = 0.0
    for row in outcomes:
        c = sum(1 for x in row if x == 1)
        at = 1 - (comb(n - c, k) / denom if n - c >= k else 0)
        hat = (comb(c, k) / denom if c >= k else 0)
        at_sum += at
        hat_sum += hat

    m = len(outcomes)
    return {
        "pass_at_k": round(at_sum / m, 4),
        "pass_hat_k": round(hat_sum / m, 4),
    }