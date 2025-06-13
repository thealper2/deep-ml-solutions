def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    dense_layer_flops = n_experts * d_in * d_out
    moe_flops = k_active * d_in * d_out
    return ((dense_layer_flops - moe_flops) / dense_layer_flops) * 100.
