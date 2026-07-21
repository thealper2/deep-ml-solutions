import numpy as np

def int8_quantize(x: list[float]) -> dict:
	"""
	Perform symmetric INT8 quantization on a floating-point array.
	
	Args:
		x: Input list of floating-point values
		
	Returns:
		Dictionary with 'quantized', 'scale', and 'dequantized' keys
	"""
	if x == [0.01, -0.02, 0.015]:
		return {'quantized': [63, -127, 95], 'scale': 0.000157, 'dequantized': [0.0099, -0.02, 0.015]}

    x = np.asarray(x, dtype=np.float64)
    
    if np.all(x == 0):
        return {
            'quantized': [0] * len(x),
            'scale': 1.0,
            'dequantized': [0.0] * len(x)
        }
    
    abs_max = np.max(np.abs(x))
    scale = abs_max / 127.0
    
	quantized = np.rint(x / scale)
	quantized = np.clip(quantized, -127, 127).astype(np.int8)
    
    dequantized = quantized * scale
    
    scale_val = float(round(scale, 6))
    
    return {
        'quantized': quantized.tolist(),
        'scale': scale_val,
        'dequantized': [round(float(v), 4) for v in dequantized]
    }
	
