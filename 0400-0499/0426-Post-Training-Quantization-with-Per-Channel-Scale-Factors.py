import numpy as np

def per_channel_quantize(weight: np.ndarray, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int array of shape (out_channels, in_features)
        - scale_factors: float array of shape (out_channels,)
        - dequantized_weights: float array of shape (out_channels, in_features)
    """
    out_channels, in_features = weight.shape
    qmax = 2 ** (bits - 1) - 1
    
    quantized_weights = np.zeros_like(weight, dtype=np.int32)
    scale_factors = np.zeros(out_channels, dtype=np.float64)
    dequantized_weights = np.zeros_like(weight, dtype=np.float64)
    
    for c in range(out_channels):
        channel_weights = weight[c, :]
        max_abs = np.max(np.abs(channel_weights))
        
        if max_abs == 0:
            scale_factors[c] = 1.0
        else:
            scale_factors[c] = max_abs / qmax
        
        quantized = np.round(channel_weights / scale_factors[c]).astype(np.int32)
        quantized = np.clip(quantized, -qmax - 1, qmax)
        quantized_weights[c, :] = quantized
        dequantized_weights[c, :] = quantized * scale_factors[c]
    
    return quantized_weights, scale_factors, dequantized_weights
