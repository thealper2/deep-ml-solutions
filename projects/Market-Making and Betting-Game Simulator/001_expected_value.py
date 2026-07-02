def expected_value(values, probabilities):
    v = np.array(values)
    p = np.array(probabilities)
    return float(np.sum(v * p))
