def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    sorted_requests = sorted(requests, key=lambda x: x['timestamp'])

    batches = []
    current_batch = []
    start_time = None

    for req in sorted_requests:
        if not current_batch:
            start_time = req['timestamp']
            current_batch.append(req)
        else:
            if (len(current_batch) >= max_batch_size or req['timestamp'] - start_time > max_wait_time):
                ids = [r['id'] for r in current_batch]
                features = [r['features'] for r in current_batch]
                proc_time = current_batch[-1]['timestamp']
                batches.append((ids, features, round(proc_time, 4)))
                current_batch = [req]
                start_time = req['timestamp']
            else:
                current_batch.append(req)

    if current_batch:
        ids = [r['id'] for r in current_batch]
        features = [r['features'] for r in current_batch]
        proc_time = current_batch[-1]['timestamp']
        batches.append((ids, features, round(proc_time, 4)))

    return batches
