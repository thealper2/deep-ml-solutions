def estimate_model_memory(num_params: int, bytes_per_param: int, optimizer: str = 'none', include_gradients: bool = False, unit: str = 'MB') -> float:
    """Estimate total memory footprint of a model.

    Args:
        num_params: number of model parameters
        bytes_per_param: bytes per parameter (e.g., 4 for fp32, 2 for fp16/bf16, 1 for fp8)
        optimizer: one of 'none', 'sgd', 'adam'
        include_gradients: whether to include a gradient buffer
        unit: 'MB' or 'GB'

    Returns:
        Memory footprint in the requested unit, rounded to 2 decimals.
    """
    memory = num_params * bytes_per_param

    if include_gradients:
        memory += num_params * bytes_per_param

    if optimizer == 'sgd':
        memory += num_params * 4
    elif optimizer == 'adam':
        memory += num_params * 8

    if unit == 'MB':
        memory /= (1024 ** 2)
    elif unit == 'GB':
        memory /= (1024 ** 3)

    return round(memory, 2)