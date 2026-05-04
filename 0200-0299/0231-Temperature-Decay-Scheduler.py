import numpy as np

def temperature_decay(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95
) -> float:
	"""
	Compute temperature at current training step using decay schedule.
	
	Temperature controls randomness in neural network outputs:
	- High temperature: More random, more exploration
	- Low temperature: More deterministic, more exploitation
	
	Args:
		schedule_type: Decay schedule type
		  'linear': Steady linear decrease
		  'exponential': Fast early decay, slow later
		  'cosine': Smooth cosine curve
		  'constant': No decay
		initial_temp: Starting temperature
		current_step: Current training step (0 to total_steps)
		total_steps: Total number of training steps
		final_temp: Minimum temperature (floor)
		decay_rate: Decay rate per step (for exponential)
	
	Returns:
		Temperature value at current step
	"""
	if schedule_type == 'constant':
		return initial_temp

	elif schedule_type == 'linear':
		progress = current_step / total_steps
		temperature = initial_temp - (initial_temp - final_temp) * progress
		return max(temperature, final_temp)

	elif schedule_type == 'exponential':
		temperature = initial_temp * (decay_rate ** current_step)
		return max(temperature, final_temp)

	elif schedule_type == 'cosine':
		progress = current_step / total_steps
		cosine_val = np.cos(np.pi * progress)
		temperature = final_temp + (initial_temp - final_temp) * (1 + cosine_val) / 2
		return max(temperature, final_temp)

	else:
		raise ValueError('Invalid schedule_type')