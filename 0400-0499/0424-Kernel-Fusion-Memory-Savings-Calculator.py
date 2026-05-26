def kernel_fusion_savings(input_elements: int, operations: list, dtype_bytes: int = 4, memory_bandwidth_gbps: float = None) -> dict:
    """
    Calculate memory traffic savings from fusing a chain of GPU kernel operations.
    
    Args:
        input_elements: Number of elements in the initial input tensor
        operations: List of dicts with 'output_elements' and optional 'extra_param_elements'
        dtype_bytes: Bytes per element (default: 4 for float32)
        memory_bandwidth_gbps: Optional GPU memory bandwidth in GB/s
    
    Returns:
        Dict with memory traffic analysis and optional timing estimates
    """
    unfused_bytes = 0
    prev_output_elements = input_elements

    for op in operations:
        output_elements = op['output_elements']
        extra_params = op.get('extra_param_elements', 0)
        unfused_bytes += (prev_output_elements + extra_params + output_elements) * dtype_bytes
        prev_output_elements = output_elements

    total_extra_params = sum(op.get('extra_param_elements', 0) for op in operations)
    fused_bytes = (input_elements + total_extra_params + operations[-1]['output_elements']) * dtype_bytes
    memory_saved_bytes = unfused_bytes - fused_bytes
    savings_percent = (memory_saved_bytes / unfused_bytes) * 100 if unfused_bytes > 0 else 0

    result = {
        'unfused_memory_bytes': unfused_bytes,
        'fused_memory_bytes': fused_bytes,
        'memory_saved_bytes': memory_saved_bytes,
        'savings_percent': round(savings_percent, 2),
    }

    if memory_bandwidth_gbps is not None:
        bandwidth_bytes_per_ms = memory_bandwidth_gbps * 1e6
        unfused_time_ms = unfused_bytes / bandwidth_bytes_per_ms
        fused_time_ms = fused_bytes / bandwidth_bytes_per_ms
        speedup = unfused_time_ms / fused_time_ms if fused_time_ms > 0 else 1.0

        result['unfused_time_ms'] = round(unfused_time_ms, 4)
        result['fused_time_ms'] = round(fused_time_ms, 4)
        result['speedup'] = round(speedup, 4)

    return result