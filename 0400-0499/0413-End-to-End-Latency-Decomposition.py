import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    for stage in stage_latencies:
        if not isinstance(stage_latencies[stage], np.ndarray):
            stage_latencies[stage] = np.array(stage_latencies[stage])
    
    n_requests = len(next(iter(stage_latencies.values())))
    stages = list(stage_latencies.keys())
    e2e_latencies = np.zeros(n_requests)
    for i in range(n_requests):
        e2e_latencies[i] = sum(stage_latencies[stage][i] for stage in stages)
    
    e2e_mean = float(np.mean(e2e_latencies))
    e2e_percentiles = {p: float(np.percentile(e2e_latencies, p)) for p in percentiles}
    stage_stats = {}
    stage_means = {}
    for stage in stages:
        data = stage_latencies[stage]
        stage_means[stage] = float(np.mean(data))
        stage_stats[stage] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'percentiles': {p: float(np.percentile(data, p)) for p in percentiles}
        }
    
    bottleneck = max(stage_means.items(), key=lambda x: x[1])[0]
    stage_pct = {stage: (stage_means[stage] / e2e_mean) * 100 for stage in stages}
    e2e_mean = round(e2e_mean, 2)
    e2e_percentiles = {p: round(v, 2) for p, v in e2e_percentiles.items()}
    
    for stage in stage_stats:
        stage_stats[stage]['mean'] = round(stage_stats[stage]['mean'], 2)
        stage_stats[stage]['std'] = round(stage_stats[stage]['std'], 2)
        stage_stats[stage]['percentiles'] = {p: round(v, 2) for p, v in stage_stats[stage]['percentiles'].items()}
    
    stage_pct = {stage: round(pct, 2) for stage, pct in stage_pct.items()}
    
    return {
        'e2e_mean': e2e_mean,
        'e2e_percentiles': e2e_percentiles,
        'stage_stats': stage_stats,
        'bottleneck': bottleneck,
        'stage_pct': stage_pct
    }