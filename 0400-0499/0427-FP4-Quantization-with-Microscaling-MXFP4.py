import numpy as np
import math

FP4_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def _quant_fp4(v):
    """Round v to the nearest FP4 E2M1 value; ties break toward smaller magnitude."""
    sign = -1.0 if v < 0 else 1.0
    a = abs(v)
    best, best_err = None, None
    for lvl in FP4_LEVELS:
        err = abs(a - lvl)
        if best is None or err < best_err - 1e-12 or (abs(err - best_err) <= 1e-12 and lvl < best):
            best, best_err = lvl, err

    return sign * best

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
	"""
	Perform MXFP4 quantization with per-block microscaling.

	Args:
		x: list of float values to quantize
		block_size: number of elements per scaling block

	Returns:
		dict with keys:
			'quantized': list of dequantized values (rounded to 4 decimals)
			'scales': list of per-block scale factors (rounded to 4 decimals)
	"""
	x = list(x)
	n = len(x)
	pad = (-n) % block_size
	xp = x + [0.0] * pad

	quantized, scales = [], []
	for start in range(0, len(xp), block_size):
		block = xp[start:start + block_size]
		amax = max(abs(v) for v in block)

		if amax == 0.0:
			scale = 1.0
		else:
			exp = math.ceil(math.log2(amax / 6.0))
			scale = 2.0 ** exp
		scales.append(round(scale, 4))

		for v in block:
			deq = _quant_fp4(v / scale) * scale
			quantized.append(round(deq, 4))

	return {"quantized": quantized[:n], "scales": scales}
