def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.
    
    Args:
        requests: List of dicts with 'arrival_time' (int) and 'tokens_needed' (int)
        max_batch_size: Maximum concurrent sequences in the batch
    
    Returns:
        Dict with 'total_time', 'avg_latency', 'avg_ttft', 'throughput'
    """
    requests = sorted(requests, key=lambda x: x['arrival_time'])
    active = []
    queue = []
    time = 0
    completed = 0
    total_latency = 0
    total_ttft = 0
    total_tokens = sum(r['tokens_needed'] for r in requests)

    req_idx = 0
    n_requests = len(requests)

    enqueued = [False] * n_requests

    while completed < n_requests:
        while req_idx < n_requests and requests[req_idx]['arrival_time'] <= time:
            queue.append(requests[req_idx])
            req_idx += 1

        while len(active) < max_batch_size and queue:
            req = queue.pop(0)
            start_time = time
            active.append({
                'remaining': req['tokens_needed'],
                'arrival': req['arrival_time'],
                'start': start_time,
            })

        for slot in active:
            slot['remaining'] -= 1

        new_active = []
        for slot in active:
            if slot['remaining'] == 0:
                completed += 1
                total_latency += (time - slot['arrival'])
                total_ttft += (slot['start'] - slot['arrival'])
            else:
                new_active.append(slot)

        active = new_active

        if not active and not queue and req_idx < n_requests:
            time = requests[req_idx]['arrival_time']
        else:
            time += 1
    
    total_time = time
    avg_latency = total_latency / n_requests
    avg_ttft = total_ttft / n_requests
    throughput = total_tokens / total_time

    return {
        'total_time': total_time,
        'avg_latency': round(avg_latency, 4),
        'avg_ttft': round(avg_ttft, 4),
        'throughput': round(throughput, 4),
    }