import math
import numpy as np


def score_with_se(outcomes):
    """{'score', 'se' (across tasks), 'seed_scores' (pass@1 per trial), 'seed_std'} for an outcome matrix."""
    arr = np.array(outcomes, dtype=float)
    N, n = arr.shape

    score = float(arr.mean())
    seed_scores = [float(arr[:, j].mean()) for j in range(n)]

    if n >= 2:
        seed_std = float(np.std(seed_scores, ddof=1))
    else:
        seed_std = None

    if N >= 2:
        row_means = arr.mean(axis=1)
        s = float(np.std(row_means, ddof=1))
        se = float(s / np.sqrt(N))
    else:
        se = None

    return {
        "score": round(score, 4),
        "se": round(float(se), 4) if se is not None else None,
        "seed_scores": [round(x, 4) for x in seed_scores],
        "seed_std": round(float(seed_std), 4) if seed_std is not None else None,
    }