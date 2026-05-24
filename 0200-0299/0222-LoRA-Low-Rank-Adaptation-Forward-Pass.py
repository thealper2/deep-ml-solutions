import numpy as np

def lora_forward(
	x: list[list[float]],
	W: list[list[float]],
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
	"""
	Compute the LoRA forward pass.
	
	Args:
		x: Input matrix (batch_size x in_features)
		W: Frozen pretrained weights (in_features x out_features)
		A: LoRA matrix A (rank x out_features)
		B: LoRA matrix B (in_features x rank)
		alpha: LoRA scaling factor
		
	Returns:
		Output matrix (batch_size x out_features)
	"""
	x = np.array(x)
	W = np.array(W)
	A = np.array(A)
	B = np.array(B)
	rank = B.shape[1]
	y_base = x @ W
	scaling = alpha / rank
	y_lora = scaling * (x @ (B @ A))
	y_final = y_base + y_lora
	return y_final.tolist()
