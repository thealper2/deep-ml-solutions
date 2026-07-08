def communication_savings(num_rounds, num_inner_steps, num_workers, param_count):
    per_event = 2 * num_workers * param_count
    diloco_scalars = num_rounds * per_event
    sync_scalars = num_rounds * num_inner_steps * per_event
    return {
        'diloco_scalars': diloco_scalars,
        'sync_scalars': sync_scalars,
        'ratio': diloco_scalars / sync_scalars,
        'savings_factor': sync_scalars / diloco_scalars,
    }
