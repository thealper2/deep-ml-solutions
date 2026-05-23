import math

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
	"""
	Compute learning rate schedule with linear warmup and cosine decay.
	
	Args:
		T: Total number of training steps
		W: Number of warmup steps
		lr_max: Maximum learning rate (reached after warmup)
		lr_min: Minimum learning rate (reached at end of training)
	
	Returns:
		List of learning rates for each step
	"""
	lrs = []

	for step in range(T):
		if step < W:
			lr = lr_max * (step / W)
		else:
			progress = (step - W) / (T - W)
			cos_val = 0.5 * (1 + math.cos(math.pi * progress))
			lr = lr_min + (lr_max - lr_min) * cos_val

		lrs.append(round(lr, 4))

	return lrs