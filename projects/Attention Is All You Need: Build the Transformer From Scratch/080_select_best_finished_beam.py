def select_best_finished_beam(finished_sequences, finished_scores, alpha):
    best_idx = 0
    best_score = float('-inf')

    for i, seq in enumerate(finished_sequences):
        length = len(seq)
        penalty = compute_length_penalty(length, alpha)
        normalized_score = finished_scores[i] / penalty
        if normalized_score > best_score:
            best_score = normalized_score
            best_idx = i

    return {
        'sequence': finished_sequences[best_idx],
        'score': best_score,
    }
