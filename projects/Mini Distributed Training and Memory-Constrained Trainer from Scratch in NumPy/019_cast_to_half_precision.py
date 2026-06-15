def cast_to_half_precision(values):
    return {k: v.copy().astype(np.float16) for k, v in values.items()}
