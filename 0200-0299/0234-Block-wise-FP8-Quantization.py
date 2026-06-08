import numpy as np

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    N = len(tensor)
    num_blocks = N // block_size
    max_fp8 = 448.0
    
    quantized = np.zeros(N, dtype=np.float32)
    scales = np.zeros(num_blocks, dtype=np.float32)
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        block = tensor[start:end]
        
        max_abs = np.max(np.abs(block))
        
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / max_fp8
        
        scales[block_idx] = scale
        
        q_block = np.round(block / scale).astype(np.int16)
        q_block = np.clip(q_block, -448.0, 448.0)
        quantized[start:end] = q_block
    
    return quantized, scales

def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    N = len(quantized)
    num_blocks = len(scales)
    
    dequantized = np.zeros(N, dtype=np.float32)
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, N)
        scale = scales[block_idx]
        dequantized[start:end] = quantized[start:end] * scale
    
    return dequantized
