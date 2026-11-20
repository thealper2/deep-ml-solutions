LETTERS = "abcdefgh"


def gold_score(prompt, completion):
    """Fraction of positions where `completion` matches `prompt` written backwards."""
    target = prompt[::-1]
    matches = 0
    for a, b in zip(completion, target):
        if a == b:
            matches += 1
    longer = max(len(completion), len(target))
    if longer == 0:
        return 0.0
    return matches / longer