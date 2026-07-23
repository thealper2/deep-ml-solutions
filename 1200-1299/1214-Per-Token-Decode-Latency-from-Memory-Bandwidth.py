def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    total_bytes = num_params * bytes_per_param
    latency_sec = total_bytes / bandwidth_bytes_per_s
    latency_ms = latency_sec * 1000
    tokens_per_sec = 1.0 / latency_sec
    return [latency_ms, tokens_per_sec]
