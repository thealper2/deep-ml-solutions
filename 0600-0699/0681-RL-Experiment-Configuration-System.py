def build_experiment_config(algorithm: str, overrides: dict = None) -> dict:
	"""
	Build a complete RL experiment configuration by merging
	user-specified overrides on top of algorithm defaults.

	Args:
		algorithm: Name of the RL algorithm (e.g., 'dqn', 'ppo', 'sarsa').
		overrides: Optional dict of parameter overrides.

	Returns:
		A dictionary with the final merged configuration sorted by keys,
		or an error dictionary if the algorithm or parameters are invalid.
	"""
	defaults = {
		"dqn": {
			"lr": 0.001,
			"gamma": 0.99,
			"epsilon": 1.0,
			"batch_size": 32,
			"buffer_size": 10000,
		},
		"ppo": {
			"lr": 0.0003,
			"gamma": 0.99,
			"clip_ratio": 0.2,
			"epochs": 10,
			"batch_size": 64,
		},
		"sarsa": {
			"lr": 0.01,
			"gamma": 0.99,
			"epsilon": 0.1,
			"batch_size": 1,
			"n_step": 1,
		}
	}

	if algorithm not in defaults:
		return {"error": "Unknown algorithm"}

	config = defaults[algorithm].copy()

	if overrides:
		for key, value in overrides.items():
			if key not in config:
				return {"error": f"Invalid parameter '{key}'"}

			config[key] = value

	config["algorithm"] = algorithm
	return dict(sorted(config.items()))