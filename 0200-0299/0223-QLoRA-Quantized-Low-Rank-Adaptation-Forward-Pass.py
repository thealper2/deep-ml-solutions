import numpy as np

def qlora_forward(
	x: list[list[float]],
	quantized_W: list[list[int]],
	scale: float,
	zero_point: float,
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
	"""
	QLoRA forward pass with 4-bit quantized frozen weights.
	
	Args:
		x: Input matrix (batch_size x in_features)
		quantized_W: 4-bit quantized weights (in_features x out_features)
		             Values are integers that need to be dequantized
		scale: Quantization scale factor
		zero_point: Quantization zero point for dequantization
		A: LoRA matrix A (rank x out_features) - full precision
		B: LoRA matrix B (in_features x rank) - full precision
		alpha: LoRA scaling factor
		
	Returns:
		Output matrix (batch_size x out_features)
	"""
	x = np.array(x)
	quantized_W = np.array(quantized_W)
	A = np.array(A)
	B = np.array(B)
	W = (quantized_W * scale) + zero_point
	rank = A.shape[0]
	scaling = alpha / rank
	frozen_path = np.dot(x, W)
	lora_path = np.dot(np.dot(x, B), A) * scaling
	combined = frozen_path + lora_path
	return combined.tolist()