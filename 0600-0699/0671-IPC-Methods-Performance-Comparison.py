def compare_ipc_methods(methods: list, message_size_bytes: int, num_messages: int) -> list:
    """
    Compare IPC methods and rank them by estimated total time.
    
    Args:
        methods: List of dicts with keys 'name', 'latency_us', 'bandwidth_mbps', 'setup_cost_us'
        message_size_bytes: Size of each message in bytes
        num_messages: Number of messages to transfer
    
    Returns:
        List of dicts with 'name', 'total_time_us', 'throughput_msgs_per_sec',
        sorted by total_time_us ascending.
    """
    results = []

    for method in methods:
        name = method['name']
        latency_us = method['latency_us']
        bandwidth_mbps = method['bandwidth_mbps']
        setup_cost_us = method['setup_cost_us']
        transfer_time_us = message_size_bytes / bandwidth_mbps
        per_msg_time_us = latency_us + transfer_time_us
        total_time_us = setup_cost_us + (num_messages * per_msg_time_us)
        throughput_msgs_per_sec = (num_messages * 1e6) / total_time_us

        results.append({
            'name': name,
            'total_time_us': round(total_time_us, 2),
            'throughput_msgs_per_sec': round(throughput_msgs_per_sec, 2),
        })

    results.sort(key=lambda x: x['total_time_us'])
    return results