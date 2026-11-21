import numpy as np


def shard_bounds(n, k):
    """Split 0..n-1 into k contiguous shards whose sizes differ by at most one."""
    q, r = divmod(n, k)
    bounds = []
    start = 0
    for i in range(k):
        size = q + 1 if i < r else q
        bounds.append((start, start + size))
        start += size

    return bounds