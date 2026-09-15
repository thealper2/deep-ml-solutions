def on_device_budget(model_mb, latency_ms, memory_budget_mb, target_fps, bits=32, sparsity=0.0, distill_ratio=1.0, speedup=None):
    if speedup is None:
        speedup = {32: 1.0, 16: 1.6, 8: 2.5, 4: 3.0}
    
    size = model_mb * distill_ratio
    lat = latency_ms * distill_ratio
    
    size = size * (1 - sparsity)
    
    size = size * (bits / 32)
    lat = lat / speedup[bits]
    
    size_mb = round(size, 3)
    latency_ms_out = round(lat, 3)
    fps = round(1000 / lat, 1)
    fits_memory = size_mb <= memory_budget_mb
    meets_fps = fps >= target_fps
    
    return {
        "size_mb": size_mb,
        "latency_ms": latency_ms_out,
        "fps": fps,
        "fits_memory": fits_memory,
        "meets_fps": meets_fps,
        "ok": fits_memory and meets_fps,
    }