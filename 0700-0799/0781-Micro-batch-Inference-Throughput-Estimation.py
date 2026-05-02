def microbatch_speedup(batch_time: float, num_microbatches: int, overhead_per_microbatch: float) -> float:
    """
    Estimate inference speedup from pipeline micro-batching.

    Args:
        batch_time: Time (seconds) to process one full batch without micro-batching.
        num_microbatches: Number of micro-batches (M >= 1).
        overhead_per_microbatch: Synchronization overhead (seconds) per micro-batch.

    Returns:
        Speedup ratio (float): batch_time / time_with_microbatching.
    """
    time = batch_time / num_microbatches + num_microbatches * overhead_per_microbatch
    speed = batch_time / time
    return round(speed, 4)
