def bucket_allreduce(grads, link=LINK, cap_mb=25.0):
    """Group gradients into buckets in backward order and schedule one all-reduce per bucket."""
    cap = cap_mb * 1e6
    buckets = []
    cur = []
    cur_bytes = 0.0

    for g in grads:
        b = g["bytes"]
        if cur and cur_bytes + b > cap:
            buckets.append(cur)
            cur = []
            cur_bytes = 0.0
        cur.append(g)
        cur_bytes += b

    if cur:
        buckets.append(cur)

    result = []
    prev_end = 0.0
    for bucket in buckets:
        bytes_total = sum(g["bytes"] for g in bucket)
        ready = max(g["ready"] for g in bucket)
        start = max(ready, prev_end)
        end = start + allreduce_us(bytes_total, link)
        result.append({
            "params": [g["name"] for g in bucket],
            "bytes": bytes_total,
            "start": start,
            "end": end,
        })
        prev_end = end

    return result