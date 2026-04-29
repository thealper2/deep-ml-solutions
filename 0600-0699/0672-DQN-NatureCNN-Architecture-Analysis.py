def nature_cnn_info(input_shape: tuple, num_actions: int) -> dict:
	"""
	Compute architecture details for the NatureCNN used in deep Q-networks.

	Args:
		input_shape: tuple of (channels, height, width)
		num_actions: number of output actions

	Returns:
		dict with architecture details
	"""
	C, H, W = input_shape
	conv1 = (H - 8) // 4 + 1
	conv1_shape = (32, conv1, conv1)
	conv1_params = 32 * C * 8 * 8 + 32

	conv2 = (conv1 - 4) // 2 + 1
	conv2_shape = (64, conv2, conv2)
	conv2_params = 64 * 32 * 4 * 4 + 64

	conv3 = (conv2 - 3) // 1 + 1
	conv3_shape = (64, conv3, conv3)
	conv3_params = 64 * 64 * 3 * 3 + 64

	flatten_size = 64 * conv3 * conv3

	fc_output_size = 512
	fc_params = flatten_size * fc_output_size + fc_output_size

	output_size = num_actions

	total_params = conv1_params + conv2_params + conv3_params + fc_params + output_size * fc_output_size + output_size

	return {
		'conv1_output_shape': conv1_shape,
		'conv2_output_shape': conv2_shape,
		'conv3_output_shape': conv3_shape,
		'flatten_size': flatten_size,
		'fc_output_size': fc_output_size,
		'output_size': output_size,
		'total_params': total_params,
	}