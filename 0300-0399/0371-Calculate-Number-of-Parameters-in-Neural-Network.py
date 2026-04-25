def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	total_parameters = 0
	for layer in layers:
		if layer['type'] == 'dense':
			layer_params = layer['input_size'] * layer['output_size']
			if layer.get('bias', -1):
				layer_params += layer['output_size']

		elif layer['type'] == 'conv2d':
			layer_params = layer['out_channels'] * layer['in_channels'] * layer['kernel_size'] * layer['kernel_size']
			if layer.get('bias', -1):
				layer_params += layer['out_channels']

		total_parameters += layer_params

	return total_parameters