import math

def compute_context_compression(
    hist_frames: int,
    height: int,
    width: int,
    latent_channels: int,
    spatial_compression: int,
    temporal_compression: int,
    spatial_downsample: int,
    temporal_downsample: int,
    dtype: str = 'fp16'
) -> dict:
    """
    Compute token counts, memory footprint, and compression ratio for
    historical context compression in an autoregressive video model.

    Returns a dict with keys: full_latent_tokens, compressed_tokens,
    compression_ratio, full_memory_bytes, compressed_memory_bytes,
    memory_saved_bytes, memory_saved_mb.
    """
    lat_t = math.ceil(hist_frames / temporal_compression)
    lat_h = math.ceil(height / spatial_compression)
    lat_w = math.ceil(width / spatial_compression)

    comp_t = math.ceil(lat_t / temporal_downsample)
    comp_h = math.ceil(lat_h / spatial_downsample)
    comp_w = math.ceil(lat_w / spatial_downsample)

    full_latent_tokens = lat_t * lat_h * lat_w
    compressed_tokens = comp_t * comp_h * comp_w

    compression_ratio = round(full_latent_tokens / compressed_tokens, 4)

    dtype_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
    }
    bytes_per_element = dtype_bytes[dtype]

    full_memory_bytes = full_latent_tokens * latent_channels * bytes_per_element
    compressed_memory_bytes = compressed_tokens * latent_channels * bytes_per_element

    memory_saved_bytes = full_memory_bytes - compressed_memory_bytes
    memory_saved_mb = round(memory_saved_bytes / (1024 * 1024), 4)

    return {
        "full_latent_tokens": full_latent_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": compression_ratio,
        "full_memory_bytes": full_memory_bytes,
        "compressed_memory_bytes": compressed_memory_bytes,
        "memory_saved_bytes": memory_saved_bytes,
        "memory_saved_mb": memory_saved_mb,
    }
