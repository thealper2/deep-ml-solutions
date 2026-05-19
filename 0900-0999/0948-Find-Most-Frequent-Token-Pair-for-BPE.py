from collections import defaultdict

def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    if not sequences:
        return None

    pair_counts = defaultdict(int)

    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1

    if not pair_counts:
        return None

    max_count = -1
    best_pair = None

    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            count = pair_counts[pair]
            if count > max_count:
                max_count = count
                best_pair = pair
            elif count == max_count and best_pair is None:
                best_pair = pair

    return best_pair