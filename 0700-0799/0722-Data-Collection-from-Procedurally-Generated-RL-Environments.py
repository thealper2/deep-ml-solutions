def collect_rollouts(seeds: list[int], grid_size: int) -> list[int]:
	"""
	Collect optimal episode returns across procedurally generated gridworlds.

	Args:
		seeds: list of integer seeds, each defining one procedural environment.
		grid_size: side length of the square grid.

	Returns:
		A list of episode returns (one per seed).
	"""
	results = []

	for seed in seeds:
		gx = seed % grid_size
		gy = (seed // grid_size) % grid_size

		if gx == 0 and gy == 0:
			gx = grid_size - 1
			gy = grid_size - 1

		steps = gx + gy
		episode_return = 10 - steps
		results.append(episode_return)

	return results