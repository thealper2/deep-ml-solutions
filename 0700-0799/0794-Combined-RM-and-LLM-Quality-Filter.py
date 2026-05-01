import math


def combined_quality_filter(rm_scores, llm_scores):
    """
    Return a boolean list marking high-quality samples by combined RM and LLM criteria.
    """
    if not rm_scores:
        return []

    n = len(rm_scores)

    sorted_rm = sorted(rm_scores)
    idx = 0.75 * (n - 1)
    idx_floor = int(math.floor(idx))
    idx_ceil = int(math.ceil(idx))

    if idx_floor == idx_ceil:
        rm_threshold = sorted_rm[idx_floor]
    else:
        weight = idx - idx_floor
        rm_threshold = sorted_rm[idx_floor] * (1 - weight) + sorted_rm[idx_ceil] * weight

    llm_max = max(llm_scores)

    selected = []
    for i in range(n):
        rm_ok = rm_scores[i] >= rm_threshold
        llm_ok = llm_scores[i] == llm_max
        selected.append(rm_ok or llm_ok)

    return selected