def lora_memory_footprint(num_params_total: int, lora_target_shapes: list, rank: int, bytes_per_element: int = 2) -> dict:
    """
    Compute GPU memory footprint (in bytes) for full fine-tuning vs low-rank
    adapter fine-tuning with the Adam optimizer.

    Args:
        num_params_total: Total number of parameters in the pre-trained model.
        lora_target_shapes: List of (d_in, d_out) tuples for adapted linear layers.
        rank: Rank r of every low-rank adapter.
        bytes_per_element: Bytes per stored element (e.g. 2 for FP16, 4 for FP32).

    Returns:
        Dict with integer fields 'full_ft_bytes', 'lora_bytes', 'lora_trainable_params'.
    """
    full_ft_bytes = num_params_total * 4 * bytes_per_element
    frozen_bytes = num_params_total * bytes_per_element
    lora_trainable_params = 0
    for d_in, d_out in lora_target_shapes:
        lora_trainable_params += (d_in * rank) + (rank * d_out)

    lora_trainable_bytes = lora_trainable_params * 4 * bytes_per_element
    lora_bytes = frozen_bytes + lora_trainable_bytes
    return {
        'full_ft_bytes': full_ft_bytes,
        'lora_bytes': lora_bytes,
        'lora_trainable_params': lora_trainable_params,
    }
