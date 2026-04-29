def progressive_batch_size(queries: list[int], milestones: list[int], batch_sizes: list[int]) -> list[int]:
	"""
	Return the active batch size at each queried token count.

	Args:
		queries: List of token counts (tokens processed so far) to query.
		milestones: Ascending token-count thresholds where the batch size steps up.
		batch_sizes: Batch sizes for each stage. Length must be len(milestones) + 1.

	Returns:
		List of batch sizes corresponding to each query.
	"""
	result = []
	n = len(milestones)

	for tokens in queries:
		if not milestones:
			batch_size = batch_sizes[0]
		elif tokens < milestones[0]:
			batch_size = batch_sizes[0]
		elif tokens >= milestones[-1]:
			batch_size = batch_sizes[-1]
		else:
			for i in range(n):
				if tokens < milestones[i]:
					batch_size = batch_sizes[i]
					break

		result.append(batch_size)

	return result