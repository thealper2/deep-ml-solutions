def analyze_canary_deployment(canary_results: list, baseline_results: list, accuracy_tolerance: float = 0.05, latency_tolerance: float = 0.10) -> dict:
    """
    Analyze canary deployment health metrics for model rollout decision.
    
    Args:
        canary_results: list of prediction results from canary (new) model
                       Each dict has 'latency_ms', 'prediction', 'ground_truth'
        baseline_results: list of prediction results from baseline (existing) model
                         Each dict has 'latency_ms', 'prediction', 'ground_truth'
        accuracy_tolerance: max acceptable relative accuracy degradation (0.05 = 5%)
        latency_tolerance: max acceptable relative latency increase (0.10 = 10%)
    
    Returns:
        dict with canary/baseline metrics and promotion recommendation
    """
    if not canary_results or not baseline_results:
        return {}
    
    canary_correct = sum(1 for r in canary_results if r['prediction'] == r['ground_truth'])
    canary_accuracy = canary_correct / len(canary_results)
    canary_avg_latency = sum(r['latency_ms'] for r in canary_results) / len(canary_results)
    
    baseline_correct = sum(1 for r in baseline_results if r['prediction'] == r['ground_truth'])
    baseline_accuracy = baseline_correct / len(baseline_results)
    baseline_avg_latency = sum(r['latency_ms'] for r in baseline_results) / len(baseline_results)
    
    if baseline_accuracy > 0:
        accuracy_change_pct = ((canary_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    else:
        accuracy_change_pct = 0.0 if canary_accuracy == 0 else 100.0
    
    if baseline_avg_latency > 0:
        latency_change_pct = ((canary_avg_latency - baseline_avg_latency) / baseline_avg_latency) * 100
    else:
        latency_change_pct = 0.0 if canary_avg_latency == 0 else 100.0
    
    accuracy_tolerance_pct = accuracy_tolerance * 100
    latency_tolerance_pct = latency_tolerance * 100 
    accuracy_acceptable = accuracy_change_pct >= -accuracy_tolerance_pct
    latency_acceptable = latency_change_pct <= latency_tolerance_pct
    promote_recommended = accuracy_acceptable and latency_acceptable
    
    return {
        'canary_accuracy': round(canary_accuracy, 4),
        'baseline_accuracy': round(baseline_accuracy, 4),
        'accuracy_change_pct': round(accuracy_change_pct, 2),
        'canary_avg_latency': round(canary_avg_latency, 2),
        'baseline_avg_latency': round(baseline_avg_latency, 2),
        'latency_change_pct': round(latency_change_pct, 2),
        'promote_recommended': promote_recommended
    }