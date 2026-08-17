import numpy as np

def apply_logits_processors(logits, generated_ids, presence_penalty,
                            frequency_penalty, repetition_penalty,
                            stop_sequences):
    """
    Apply repetition / frequency / presence penalties and detect stop suffixes.

    Returns (adjusted_logits_list, should_stop).
    """
    counts = {}
    for token in generated_ids:
        counts[token] = counts.get(token, 0) + 1

    adjusted = list(logits)
    V = len(logits)

    for token in range(V):
        if token in counts:
            c = counts[token]
            if adjusted[token] > 0:
                adjusted[token] /= repetition_penalty
            else:
                adjusted[token] *= repetition_penalty

            adjusted[token] -= presence_penalty + frequency_penalty * c
        
    adjusted = [round(val, 4) for val in adjusted]

    should_stop = False
    for stop_seq in stop_sequences:
        if not stop_seq:
            continue
        
        if len(generated_ids) >= len(stop_seq):
            suffix = generated_ids[-len(stop_seq):]
            if suffix == stop_seq:
                should_stop = True
                break

    return adjusted, should_stop