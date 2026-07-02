def one_reroll_die_value(sides):
    probs = [1 / sides] * sides
    reroll_ev = expected_value(range(1, sides + 1), probs)
    reroll_faces = [face for face in range(1, sides + 1) if face < reroll_ev]
    optimal_values = [reroll_ev if face < reroll_ev else face for face in range(1, sides + 1)]
    value = expected_value(optimal_values, probs)

    return {
        'value': value,
        'reroll_faces': reroll_faces,
    }
