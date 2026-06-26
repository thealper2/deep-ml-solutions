def serialize_q_table_to_dict(q_table):
    """Convert a Q-table (str -> np.ndarray shape (9,)) into a plain dict (str -> list of floats)."""
    result = {}
    for key, value in q_table.items():
        if isinstance(value, np.ndarray):
            result[key] = value.astype(float).tolist()
        else:
            result[key] = float(value)

    return result
