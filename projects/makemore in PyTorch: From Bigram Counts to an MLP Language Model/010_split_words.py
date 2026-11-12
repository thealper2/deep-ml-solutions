import random

def split_words(words, seed):
    """Shuffle a copy with random.Random(seed); return the 80/10/10 (train, dev, test) split."""
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]