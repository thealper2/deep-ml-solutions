import random


def make_preference_pairs(n, seed, length=4, p_copy=0.6, p_typo=0.05):
    """n tuples (prompt, chosen, rejected); ties in gold score are discarded."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        prompt = "".join(LETTERS[rng.randrange(len(LETTERS))] for _ in range(length))
        a = write_answer(prompt, rng, p_copy, p_typo)
        b = write_answer(prompt, rng, p_copy, p_typo)
        sa = gold_score(prompt, a)
        sb = gold_score(prompt, b)
        if sa == sb:
            continue
        if sa > sb:
            pairs.append((prompt, a, b))
        else:
            pairs.append((prompt, b, a))
    return pairs