def evaluate_preemption(
    num_tokens_list: list,
    kv_cache_bytes_per_token: int,
    swap_bandwidth_gbps: float,
    base_flops_per_token: int,
    attn_flops_per_token_pair: int,
    compute_tflops: float
) -> dict:
    """
    Evaluate preemption strategies (swap vs recompute) for a set of requests.

    Args:
        num_tokens_list: Number of tokens processed so far for each request
        kv_cache_bytes_per_token: Bytes of KV cache stored per token
        swap_bandwidth_gbps: GPU-CPU memory bandwidth in GB/s
        base_flops_per_token: FLOPs for non-attention layers per token during recomputation
        attn_flops_per_token_pair: FLOPs for attention per token pair during recomputation
        compute_tflops: GPU compute throughput in TFLOPS

    Returns:
        Dictionary with strategies, costs, and total cost
    """
    strategies = []
    swap_costs = []
    recompute_costs = []
    total = 0.0

    for n in num_tokens_list:
        swap_ms = 2 * n * kv_cache_bytes_per_token / (swap_bandwidth_gbps * 1e9) * 1000
        recompute_ms = (n * base_flops_per_token + n * n * attn_flops_per_token_pair) / (compute_tflops * 1e12) * 1000

        if swap_ms <= recompute_ms:
            strategy = 'swap'
            chosen = swap_ms
        else:
            strategy = 'recompute'
            chosen = recompute_ms

        strategies.append(strategy)
        s_round = round(swap_ms, 4)
        r_round = round(recompute_ms, 4)
        swap_costs.append(s_round)
        recompute_costs.append(r_round)
        total += s_round if strategy == 'swap' else r_round

    return {
        'strategies': strategies,
        'swap_costs_ms': swap_costs,
        'recompute_costs_ms': recompute_costs,
        'total_cost_ms': round(total, 4),
    }