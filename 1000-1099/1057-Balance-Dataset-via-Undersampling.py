def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if not data:
        return []

    freq = {}
    for i in data:
        sample, label = i
        freq[label] = freq.get(label, 0) + 1

    min_count = min(freq.values())
    freq = {k: 0 for k in freq.keys()}
    balanced = []
    
    for i in data:
        sample, label = i
        if freq[label] < min_count:
            balanced.append(i)
            freq[label] += 1
        else:
            continue

    return balanced