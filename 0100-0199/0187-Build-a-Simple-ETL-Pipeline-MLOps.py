def run_etl(csv_text: str) -> list[tuple[str, float]]:
	"""
	Run a simple ETL pipeline over CSV text with header user_id,event_type,value.

	Returns a sorted list of (user_id, total_value) for event_type == "purchase".
	"""
	lines = csv_text.strip().split('\n')

	if not lines:
		return []

	data_lines = lines[1:] if lines[0].startswith('user_id') else lines
	user_totals = {}

	for line in data_lines:
		line = line.strip()
		if not line:
			continue

		parts = [p.strip() for p in line.split(',')]
		if len(parts) < 3:
			continue

		user_id, event_type, value_str = parts[0], parts[1], parts[2]

		if event_type != 'purchase':
			continue

		try:
			value = float(value_str)
		except ValueError:
			continue
		
		user_totals[user_id] = user_totals.get(user_id, 0.0) + value

	result = [(user_id, total) for user_id, total in sorted(user_totals.items())]
	return result