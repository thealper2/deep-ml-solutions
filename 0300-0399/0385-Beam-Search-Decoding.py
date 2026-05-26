import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    beams = [([], 0.0)]
    for t in range(log_probs.shape[0]):
        candidates = []
        for seq, score in beams:
            for token in range(log_probs.shape[1]):
                new_score = score + log_probs[t, token]
                candidates.append((seq + [token], new_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    return beams[:top_k]