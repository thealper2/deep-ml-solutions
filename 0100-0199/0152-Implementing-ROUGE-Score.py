import numpy as np

def rouge_1_score(reference: str, candidate: str, n: int = 1) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    if len(ref_words) < n or len(cand_words) < n:
        return 0.0, 0.0, 0.0
    
    get_ngrams = lambda words, n: [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    ref_ngrams = get_ngrams(ref_words, n)
    cand_ngrams = get_ngrams(cand_words, n)

    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1

    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    matches = 0
    for ng, count in cand_counts.items():
        matches += min(count, ref_counts.get(ng, 0))

    precision = matches / len(cand_ngrams) if len(cand_ngrams) > 0 else 0
    recall = matches / len(ref_ngrams) if len(ref_ngrams) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {'precision': precision, 'recall': recall, 'f1': f1}