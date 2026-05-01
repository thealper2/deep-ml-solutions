import numpy as np

def average_checkpoints(checkpoints: list) -> dict:
    if not checkpoints:
        return {}

    d = {}
    for checkpoint in checkpoints:
        for key, value in checkpoint.items():
            if key in d:
                d[key].append(value)
            else:
                d[key] = [value]

    result = {}
    for key, value in d.items():
        result[key] = np.mean(value, axis=0).tolist()

    return result