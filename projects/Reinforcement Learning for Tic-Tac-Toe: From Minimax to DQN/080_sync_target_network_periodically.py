import numpy as np

def sync_target_network_periodically(online_params, target_params, step_count, sync_every_k):
    """Copy online -> target every sync_every_k steps; otherwise leave target unchanged."""
    if step_count > 0 and step_count % sync_every_k == 0:
        return build_target_network_copy(online_params)

    return target_params
