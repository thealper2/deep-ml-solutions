def mig_resource_allocation(gpu_config: dict, mig_profiles: list, workloads: list) -> dict:
    """
    Allocate MIG instances to workloads on a single GPU.
    
    Args:
        gpu_config: Dict with 'total_compute_slices' and 'total_memory_gb'
        mig_profiles: List of available MIG profile dicts
        workloads: List of workload requirement dicts
    
    Returns:
        Dict with allocation results and utilization metrics
    """
    workloads_sorted = sorted(workloads, key=lambda w: (-w['min_compute_slices'], -w['min_memory_gb']))
    profiles_sorted = sorted(mig_profiles, key=lambda p: (p['compute_slices'], p['memory_gb']))

    remaining_compute = gpu_config['total_compute_slices']
    remaining_memory = gpu_config['total_memory_gb']

    allocations = []
    rejected = []

    for workload in workloads_sorted:
        best_profile = None
        for profile in profiles_sorted:
            if (profile['compute_slices'] >= workload['min_compute_slices'] and
                profile['memory_gb'] >= workload['min_memory_gb'] and
                profile['compute_slices'] <= remaining_compute and
                profile['memory_gb'] <= remaining_memory):
                best_profile = profile
                break

        if best_profile is None:
            rejected.append(workload['name'])
        else:
            allocations.append({
                'workload': workload['name'],
                'profile': best_profile['name'],
                'compute_slices': best_profile['compute_slices'],
                'memory_gb': best_profile['memory_gb'],
            })
            remaining_compute -= best_profile['compute_slices']
            remaining_memory -= best_profile['memory_gb']

    total_compute_used = gpu_config['total_compute_slices'] - remaining_compute
    total_memory_used = gpu_config['total_memory_gb'] - remaining_memory

    compute_utilization = (total_compute_used / gpu_config['total_compute_slices']) * 100
    memory_utilization = (total_memory_used / gpu_config['total_memory_gb']) * 100

    return {
        'allocations': allocations,
        'total_compute_used': total_compute_used,
        'total_memory_used': round(total_memory_used, 2),
        'compute_utilization': round(compute_utilization, 2),
        'memory_utilization': round(memory_utilization, 2),
        'workloads_served': len(allocations),
        'workloads_rejected': rejected,
    }