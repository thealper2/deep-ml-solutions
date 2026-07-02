def update_remaining_card_value(remaining_counts, revealed_value):
    new_counts = remaining_counts.copy()

    if revealed_value in new_counts:
        new_counts[revealed_value] -= 1
        if new_counts[revealed_value] == 0:
            del new_counts[revealed_value]

    if not new_counts:
        expected_val = 0.0
    else:
        values = list(new_counts.keys())
        probabilities = [count / sum(new_counts.values()) for count in new_counts.values()]
        expected_val = expected_value(values, probabilities)

    return {
        'remaining_counts': new_counts,
        'expected_value': expected_val,
    }
