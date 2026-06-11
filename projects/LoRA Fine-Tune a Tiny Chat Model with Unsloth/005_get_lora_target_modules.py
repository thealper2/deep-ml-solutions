def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
