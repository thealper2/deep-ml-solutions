def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    
    Args:
        timestamps_ms: List of N sorted timestamps (ms) marking interval boundaries.
        gpu_util_pct: List of N-1 GPU utilization percentages (0-100) per interval.
        phase_labels: List of N-1 phase labels per interval.
    
    Returns:
        dict with keys: total_time_ms, avg_gpu_util_pct, phase_stats,
                        bottleneck_phase, gpu_idle_fraction_pct
    """
    total_time = timestamps_ms[-1] - timestamps_ms[0]

    total_util = 0.0
    durations = []
    for i in range(len(gpu_util_pct)):
        duration = timestamps_ms[i + 1] - timestamps_ms[i]
        durations.append(duration)
        total_util += gpu_util_pct[i] * duration

    avg_util = total_util / total_time if total_time > 0 else 0

    phase_data = {}
    for i, phase in enumerate(phase_labels):
        if phase not in phase_data:
            phase_data[phase] = {
                'total_time_ms': 0.0,
                'total_util': 0.0,
            }

        phase_data[phase]['total_time_ms'] += durations[i]
        phase_data[phase]['total_util'] += gpu_util_pct[i] * durations[i]

    phase_stats = {}
    for phase in sorted(phase_data.keys()):
        total_time_phase = phase_data[phase]['total_time_ms']
        total_util_phase = phase_data[phase]['total_util']

        avg_util_phase = total_util_phase / total_time_phase if total_time_phase > 0 else 0
        time_fraction = (total_time_phase / total_time) * 100 if total_time > 0 else 0

        phase_stats[phase] = {
            'total_time_ms': round(total_time_phase, 2),
            'avg_util_pct': round(avg_util_phase, 2),
            'time_fraction_pct': round(time_fraction, 2),
        }

    bottleneck_phase = None
    max_wasted = -1
    earliest_first_interval = float('inf')

    for i, phase in enumerate(phase_labels):
        wasted = durations[i] * (100 - gpu_util_pct[i])
        if wasted > max_wasted or (wasted == max_wasted and i < earliest_first_interval):
            max_wasted = wasted
            bottleneck_phase = phase
            earliest_first_interval = i

        
    idle_fraction = 100 - avg_util

    return {
        'total_time_ms': round(total_time, 2),
        'avg_gpu_util_pct': round(avg_util, 2),
        'phase_stats': phase_stats,
        'bottleneck_phase': bottleneck_phase,
        'gpu_idle_fraction_pct': round(idle_fraction, 2),
    }