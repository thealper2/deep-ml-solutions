def roofline_analysis(peak_gflops: float, peak_bandwidth_gbs: float, operations: list) -> dict:
	"""
	Perform Roofline Model analysis for GPU operations.

	Args:
		peak_gflops: Peak compute throughput in GFLOPS
		peak_bandwidth_gbs: Peak memory bandwidth in GB/s
		operations: List of dicts with keys 'name', 'flops', 'bytes'

	Returns:
		Dict with 'ridge_point' and 'operations' list containing
		per-operation analysis results.
	"""
	ridge_point = peak_gflops / peak_bandwidth_gbs
	results = []

	for op in operations:
		oi = op['flops'] / op['bytes']

		if oi >= ridge_point:
			bottleneck = 'compute-bound'
			attainable_gflops = peak_gflops
		else:
			bottleneck = 'memory-bound'
			attainable_gflops = peak_bandwidth_gbs * oi

		efficiency = (attainable_gflops / peak_gflops) * 100

		results.append({
			'name': op['name'],
			'operational_intensity': round(oi, 6),
			'attainable_gflops': round(attainable_gflops, 6),
			'bottleneck': bottleneck,
			'efficiency': round(efficiency, 6),
		})

	return {
		'ridge_point': round(ridge_point, 6),
		'operations': results,
	}