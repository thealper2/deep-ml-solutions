def profile_env_wrappers(wrapper_names: list, cumulative_step_times: list) -> dict:
    """
    Profile RL environment wrappers to identify per-wrapper overhead.
    
    Args:
        wrapper_names: List of wrapper names from outermost to innermost.
        cumulative_step_times: List of lists of timing measurements (ms) per wrapper level.
    
    Returns:
        dict with keys: 'wrappers' (list of per-wrapper stats), 'bottleneck' (str),
                        'total_overhead_ms' (float)
    """
    mean_cumulative = []
    for times in cumulative_step_times:
        mean_cumulative.append(sum(times) / len(times))

    wrappers = []
    max_overhead = -1
    bottleneck = None

    for i, name in enumerate(wrapper_names):
        if i == len(wrapper_names) - 1:
            overhead_ms = mean_cumulative[i]
        else:
            overhead_ms = mean_cumulative[i] - mean_cumulative[i + 1]

        total_time = mean_cumulative[0]
        if total_time > 0:
            overhead_pct = (overhead_ms / total_time) * 100
        else:
            overhead_pct = 0.0

        wrappers.append({
            'name': name,
            'mean_cumulative_ms': round(mean_cumulative[i], 2),
            'overhead_ms': round(overhead_ms, 2),
            'overhead_pct': round(overhead_pct, 2),
        })

        if overhead_ms > max_overhead:
            max_overhead = overhead_ms
            bottleneck = name

    return {
        'wrappers': wrappers,
        'bottleneck': bottleneck,
        'total_overhead_ms': round(mean_cumulative[0], 2),
    }