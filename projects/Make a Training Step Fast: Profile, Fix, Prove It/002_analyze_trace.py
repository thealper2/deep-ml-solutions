def _union_length(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total

def analyze_trace(trace, launch_us=5.0, top=3):
    """Busy and idle time, data wait, exposed communication, top kernel groups and a verdict."""
    if not trace:
        return {
            "step_us": 0.0,
            "gpu_busy_us": 0.0,
            "gpu_idle_share": 0.0,
            "data_wait_us": 0.0,
            "exposed_comm_us": 0.0,
            "small_kernels": 0,
            "top_groups": [],
            "verdict": "compute_bound",
        }

    first_start = min(e["start"] for e in trace)
    last_end = max(e["end"] for e in trace)
    step_us = float(last_end - first_start)

    kernel_ivs = [(e["start"], e["end"]) for e in trace if e["kind"] == "kernel"]
    coll_ivs = [(e["start"], e["end"]) for e in trace if e["kind"] == "collective"]

    gpu_busy_us = float(_union_length(kernel_ivs))
    combined = _union_length(kernel_ivs + coll_ivs)
    exposed_comm_us = float(combined - gpu_busy_us)

    data_wait_us = float(sum(e["end"] - e["start"] for e in trace if e["kind"] == "dataloader_wait"))

    small_kernels = sum(
        1 for e in trace if e["kind"] == "kernel" and (e["end"] - e["start"]) < launch_us
    )

    groups = {}
    for e in trace:
        if e["kind"] == "kernel":
            g = e.get("group")
            groups[g] = groups.get(g, 0.0) + (e["end"] - e["start"])
    top_groups = sorted(
        ((g, round(t, 3)) for g, t in groups.items()),
        key=lambda x: (-x[1], x[0]),
    )[:top]

    gpu_idle_share = float((step_us - gpu_busy_us) / step_us) if step_us > 0 else 0.0

    if step_us > 0 and data_wait_us >= 0.2 * step_us:
        verdict = "input_bound"
    elif step_us > 0 and exposed_comm_us >= 0.2 * step_us:
        verdict = "communication_bound"
    elif gpu_idle_share >= 0.2:
        verdict = "launch_bound"
    else:
        verdict = "compute_bound"

    return {
        "step_us": step_us,
        "gpu_busy_us": gpu_busy_us,
        "gpu_idle_share": gpu_idle_share,
        "data_wait_us": data_wait_us,
        "exposed_comm_us": exposed_comm_us,
        "small_kernels": small_kernels,
        "top_groups": top_groups,
        "verdict": verdict,
    }