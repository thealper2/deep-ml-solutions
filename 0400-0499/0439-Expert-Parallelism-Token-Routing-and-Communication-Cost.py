def expert_parallel_comm_cost(num_devices: int, num_experts: int, tokens_per_device: list, token_size_bytes: int, bandwidth_gbps: float = None) -> dict:
    """
    Analyze communication costs for expert-parallel MoE token routing.
    
    Args:
        num_devices: Number of devices (GPUs)
        num_experts: Total number of experts (evenly divided across devices)
        tokens_per_device: tokens_per_device[d] is a list of expert IDs for tokens on device d
        token_size_bytes: Bytes per token embedding
        bandwidth_gbps: Optional interconnect bandwidth in GB/s
    
    Returns:
        Dict with dispatch matrix, communication costs, and load analysis
    """
    E = num_experts // num_devices
    
    expert_to_device = {}
    for d in range(num_devices):
        for e in range(d * E, (d + 1) * E):
            expert_to_device[e] = d

    dispatch_matrix = [[0] * num_devices for _ in range(num_devices)]
    load_per_device = [0] * num_devices

    for src_device in range(num_devices):
        for expert_id in tokens_per_device[src_device]:
            dst_device = expert_to_device[expert_id]
            dispatch_matrix[src_device][dst_device] += 1
            load_per_device[dst_device] += 1

    total_comm_bytes = 0
    max_device_send_bytes = 0
    max_device_recv_bytes = 0

    for i in range(num_devices):
        send_bytes = 0
        recv_bytes = 0
        for j in range(num_devices):
            if i != j:
                tokens = dispatch_matrix[i][j]
                bytes_send = tokens * token_size_bytes
                send_bytes += bytes_send
                total_comm_bytes += bytes_send

                tokens_received = dispatch_matrix[j][i]
                recv_bytes += tokens_received * token_size_bytes

        max_device_send_bytes = max(max_device_send_bytes, send_bytes)
        max_device_recv_bytes = max(max_device_recv_bytes, recv_bytes)

    avg_load = sum(load_per_device) / num_devices
    max_load = max(load_per_device)
    load_imbalance = round(max_load / avg_load, 4) if avg_load > 0 else 1.0

    result = {
        'dispatch_matrix': dispatch_matrix,
        'total_comm_bytes': total_comm_bytes,
        'max_device_send_bytes': max_device_send_bytes,
        'max_device_recv_bytes': max_device_recv_bytes,
        'load_per_device': load_per_device,
        'load_imbalance': load_imbalance,
    }

    if bandwidth_gbps is not None:
        bandwidth_bytes_per_ms = bandwidth_gbps * 1e6
        bottleneck_bytes = max(max_device_send_bytes, max_device_recv_bytes)
        comm_time_ms = round(bottleneck_bytes / bandwidth_bytes_per_ms, 4)
        result['comm_time_ms'] = comm_time_ms

    return result
