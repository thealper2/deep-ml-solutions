def compute_mfu(num_params: float, throughput_tokens_per_sec: float, peak_flops: float) -> float:
    """
    Args:
        num_params: number of model parameters
        throughput_tokens_per_sec: achieved training throughput (tokens/sec)
        peak_flops: theoretical peak FLOPS of the hardware
    Returns:
        MFU as a percentage (0-100), rounded to 4 decimals
    """
    achieved_flops = 6 * num_params * throughput_tokens_per_sec
    mfu = (achieved_flops / peak_flops)
    return round(mfu, 4)