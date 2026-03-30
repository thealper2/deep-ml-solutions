def analyze_memory_fragmentation(block_status: list) -> dict:
	"""
	Analyze memory fragmentation in a block-based memory pool.

	Args:
		block_status: List of ints where 1 = allocated block, 0 = free block

	Returns:
		dict with keys:
			'utilization': float, fraction of allocated blocks
			'num_free_fragments': int, count of contiguous free regions
			'largest_free_fragment': int, size of largest contiguous free region
			'fragmentation_ratio': float, measure of free memory scatter
	"""
	total_blocks = len(block_status)
	allocated = sum(block_status)
	free_blocks = total_blocks - allocated

	utilization = allocated / total_blocks if total_blocks > 0 else 0.0

	num_free_fragments = 0
	largest_free_fragment = 0
	current_fragment = 0

	for i, block in enumerate(block_status):
		if block == 0:
			current_fragment += 1
			largest_free_fragment = max(largest_free_fragment, current_fragment)
		else:
			if current_fragment > 0:
				num_free_fragments += 1
				current_fragment = 0

	if current_fragment > 0:
		num_free_fragments += 1

	if free_blocks == 0:
		fragmentation_ratio = 0.0
	else:
		fragmentation_ratio = 1 - (largest_free_fragment / free_blocks)

	return {
		"utilization": round(utilization, 4),
		"num_free_fragments": num_free_fragments,
		"largest_free_fragment": largest_free_fragment,
		"fragmentation_ratio": round(fragmentation_ratio, 4),
	}
