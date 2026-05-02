import numpy as np
from collections import Counter

def kl_divergence_filter(reference_freq: dict, documents: list, threshold: float, alpha: float = 1.0) -> dict:
    """Filter documents by KL divergence from a reference token distribution."""
    kl_divs = []
    flagged = []
    kept = []

    for doc in documents:
        doc_counts = Counter(doc)
        V = list(doc_counts.keys())
        V_size = len(V)

        total_doc = sum(doc_counts.values())
        P = {t: doc_counts[t] / total_doc for t in V}

        ref_total = sum(reference_freq.get(t, 0) for t in V)
        denom = ref_total + alpha * V_size
        Q = {t: (reference_freq.get(t, 0) + alpha) / denom for t in V}

        kl = 0.0
        for t in V:
            p = P[t]
            q = Q[t]
            if p > 0:
                kl += p * np.log(p / q)

        kl = round(float(kl), 4)
        kl_divs.append(kl)

        is_flagged = kl > threshold
        flagged.append(is_flagged)
        if not is_flagged:
            kept.append(doc)

    return {
        'kl_divergences': kl_divs,
        'flagged': flagged,
        'kept': kept
    }