def kv_cache_offload_sim(tiers: list[dict], entries: list[dict], recompute_latency_us: float = 1000.0) -> dict:
    """
    Simulate tiered KV cache offloading for LLM inference.

    Args:
        tiers: list of dicts with keys 'name' (str), 'capacity' (int), 'latency_us' (float)
        entries: list of dicts with keys 'id' (int), 'recency' (float)
        recompute_latency_us: latency penalty for evicted entries

    Returns:
        dict with 'placement', 'evicted', 'total_latency_us', 'avg_latency_us'
    """
    if not entries:
        return {
            "placement": {t["name"]: [] for t in sorted(tiers, key=lambda x: x["latency_us"])},
            "evicted": [],
            "total_latency_us": 0.0,
            "avg_latency_us": 0.0,
        }

    ordered = sorted(tiers, key=lambda t: t["latency_us"])
    sorted_entries = sorted(entries, key=lambda e: (-e["recency"], e["id"]))

    placement = {t["name"]: [] for t in ordered}
    evicted = []

    idx = 0
    for tier in ordered:
        remaining = tier["capacity"]
        while remaining > 0 and idx < len(sorted_entries):
            placement[tier["name"]].append(sorted_entries[idx]["id"])
            idx += 1
            remaining -= 1

    while idx < len(sorted_entries):
        evicted.append(sorted_entries[idx]["id"])
        idx += 1

    for name in placement:
        placement[name].sort()
    evicted.sort()

    total = 0.0
    for tier in ordered:
        total += len(placement[tier["name"]]) * tier["latency_us"]
    total += len(evicted) * recompute_latency_us

    count = len(entries)
    return {
        "placement": placement,
        "evicted": evicted,
        "total_latency_us": round(total, 2),
        "avg_latency_us": round(total / count, 2),
    }