def alpha_beta_pruning(tree, is_maximizing=True, alpha: float = float('-inf'), beta: float = float('inf')):
	"""
	Perform alpha-beta pruning on a game tree.

	Args:
		tree: A nested list representing the game tree. Leaf nodes are numbers,
		      internal nodes are lists of children.
		is_maximizing: Whether the root node is a maximizing player.

	Returns:
		A tuple (value, nodes_evaluated) where value is the optimal minimax
		value and nodes_evaluated is the number of leaf nodes examined.
	"""
	if isinstance(tree, (int, float)):
		return tree, 1

	if is_maximizing:
		value = float('-inf')
		total_leaves = 0
		for child in tree:
			child_val, leaves = alpha_beta_pruning(child, False, alpha, beta)
			total_leaves += leaves
			value = max(value, child_val)
			alpha = max(alpha, value)
			if alpha >= beta:
				break

		return value, total_leaves

	else:
		value = float('inf')
		total_leaves = 0
		for child in tree:
			child_val, leaves = alpha_beta_pruning(child, True, alpha,beta)
			total_leaves += leaves
			value = min(value, child_val)
			beta = min(beta, value)
			if alpha >= beta:
				break

		return value, total_leaves