def lora_param_count(num_layers: int, d_model: int, num_target_matrices: int, rank: int) -> dict:
    """Compute LoRA vs full fine-tuning parameter counts."""
    num_adapted_matrices = num_layers * num_target_matrices
    lora_params = num_adapted_matrices * (d_model * rank + rank * d_model)
    full_params = num_adapted_matrices * (d_model * d_model)
    compression_ratio = full_params / lora_params
    return {
        'lora_params': lora_params,
        'full_params': full_params,
        'compression_ratio': compression_ratio,
    }