def apply_causal_mask(scores, query_offset=0):
    Tq, Tk = scores.shape
    mask = np.zeros_like(scores)
    for i in range(Tq):
        abs_pos = query_offset + i
        mask[i, :abs_pos + 1] = 1.0

    scores = np.where(mask == 1.0, scores, -np.inf)
    return scores
