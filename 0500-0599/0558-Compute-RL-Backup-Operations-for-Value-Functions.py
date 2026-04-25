def rl_backup(backup_type: str, transitions, gamma: float, values: dict, policy: dict = None) -> float:
	"""
	Compute the result of an RL backup operation.

	Args:
		backup_type: One of 'action_value', 'state_value', 'optimal_value'
		transitions: For 'action_value': list of (prob, next_state, reward).
		             For 'state_value'/'optimal_value': dict {action: [(prob, next_state, reward), ...]}.
		gamma: Discount factor
		values: Dict mapping state -> current value estimate
		policy: Dict mapping action -> probability (only for 'state_value')

	Returns:
		Backed-up value rounded to 4 decimal places
	"""
	def compute_q_value(action_transitions):
		q = 0.0
		for prob, next_state, reward in action_transitions:
			v_next = values.get(next_state, 0.0)
			q += prob * (reward + gamma * v_next)

		return q

	if backup_type == 'action_value':
		q = compute_q_value(transitions)
		return round(q, 4)

	elif backup_type == 'state_value':
		v = 0.0
		for action, action_transitions in transitions.items():
			action_prob = policy.get(action, 0.0)
			q = compute_q_value(action_transitions)
			v += action_prob * q

		return round(v, 4)

	elif backup_type == 'optimal_value':
		best_q = float('-inf')
		for action, action_transitions in transitions.items():
			q = compute_q_value(action_transitions)
			best_q = max(best_q, q)

		return round(best_q, 4)

	else:
		raise ValueError(f"Unknown backup type: {backup_type}")