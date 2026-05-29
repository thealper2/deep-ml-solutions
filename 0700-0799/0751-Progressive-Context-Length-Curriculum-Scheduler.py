def progressive_batch_sizes(steps, milestones, token_budget):
    """
    Compute per-step batch sizes under a progressive context-length curriculum.

    Args:
        steps: list[int] of training step indices to query.
        milestones: list of [start_step, seq_len], sorted ascending by start_step,
                    with the first entry having start_step == 0.
        token_budget: int, target tokens per batch.

    Returns:
        list[int]: batch size at each queried step.
    """
    milestones.sort(key=lambda x: x[0])
    batch_sizes = []
    for step in steps:
        active_len = milestones[0][1]
        for start_step, seq_len in milestones:
            if step < start_step:
                break
            
            active_len = seq_len

        batch_size = token_budget // active_len
        if batch_size < 1:
            batch_size = 1

        batch_sizes.append(batch_size)

    return batch_sizes
