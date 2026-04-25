import numpy as np

def target_network_update(
    online_params: list,
    target_params: list,
    tau: float = 0.005,
    update_type: str = 'soft'
) -> list:
    """
    Perform target network parameter update.
    
    Args:
        online_params: List of numpy arrays representing online network weights
        target_params: List of numpy arrays representing target network weights
        tau: Soft update interpolation coefficient (0 < tau <= 1)
        update_type: 'soft' for gradual blending, 'hard' for direct copy
    
    Returns:
        List of updated target network parameter arrays
    """
    updated_target = []

    if update_type == 'soft':
        for online, target in zip(online_params, target_params):
            updated = tau * online + (1 - tau) * target
            updated_target.append(updated)

    elif update_type == 'hard':
        for online in online_params:
            updated_target.append(online.copy())

    else:
        raise ValueError("Unknown update type")

    return updated_target