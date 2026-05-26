import math

def multi_gpu_comm_overhead(message_size_gb: float, num_gpus: int,
                           nvlink_config: dict, ib_config: dict,
                           operation: str) -> dict:
    """
    Compare communication overhead between NVLink and InfiniBand
    for collective GPU operations.
    
    Args:
        message_size_gb: Data payload size in GB
        num_gpus: Number of GPUs
        nvlink_config: Dict with 'bandwidth_gbps' and 'latency_us'
        ib_config: Dict with 'bandwidth_gbps' and 'latency_us'
        operation: One of 'all_reduce', 'all_gather', 'broadcast'
    
    Returns:
        Dict with timing comparison and bottleneck analysis
    """
    def compute_time(bandwidth_gbps, latency_us):
        latency_ms = latency_us / 1000

        if operation == 'all_reduce':
            num_steps = 2 * (num_gpus - 1)
            data_per_step = message_size_gb / num_gpus
        elif operation == 'all_gather':
            num_steps = num_gpus - 1
            data_per_step = message_size_gb / num_gpus
        elif operation == 'broadcast':
            num_steps = math.ceil(math.log2(num_gpus))
            data_per_step = message_size_gb
        else:
            raise ValueError('Unknown operation')

        bandwidth_time_ms = (data_per_step / bandwidth_gbps) * 1000 * num_steps
        latency_time_ms = latency_ms * num_steps
        total_time_ms = bandwidth_time_ms + latency_time_ms

        if bandwidth_time_ms >= latency_time_ms:
            bottleneck = 'bandwidth'
        else:
            bottleneck = 'latency'

        return total_time_ms, bottleneck
    
    nvlink_time, nvlink_bottleneck = compute_time(
        nvlink_config['bandwidth_gbps'],
        nvlink_config['latency_us'],
    )

    ib_time, ib_bottleneck = compute_time(
        ib_config['bandwidth_gbps'],
        ib_config['latency_us'],
    )

    speedup = ib_time / nvlink_time if nvlink_time > 0 else 0

    return {
        'nvlink_time_ms': round(nvlink_time, 4),
        'ib_time_ms': round(ib_time, 4),
        'speedup': round(speedup, 4),
        'nvlink_bottleneck': nvlink_bottleneck,
        'ib_bottleneck': ib_bottleneck,
    }