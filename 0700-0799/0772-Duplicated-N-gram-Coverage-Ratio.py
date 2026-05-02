def dup_ngram_ratio(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))

    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1

    duplicated_count = sum(cnt for cnt in counts.values() if cnt > 1)
    total_ngrams = len(ngrams)
    ratio = duplicated_count / total_ngrams
    return round(ratio, 4)