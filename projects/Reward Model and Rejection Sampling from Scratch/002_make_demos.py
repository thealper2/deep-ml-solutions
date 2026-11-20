import random


def write_answer(prompt, rng, p_copy=0.6, p_typo=0.05):
    """One answer from a writer who, at each position, copies the prompt's letter with probability p_copy and mistypes with probability p_typo."""
    n = len(prompt)
    out = []
    for i in range(n):
        if rng.random() < p_copy:
            letter = prompt[i]
        else:
            letter = prompt[n - 1 - i]
        if rng.random() < p_typo:
            letter = LETTERS[rng.randrange(len(LETTERS))]
        out.append(letter)
    return "".join(out)


def make_demos(n, seed, length=4, p_copy=0.6, p_typo=0.05):
    """n (prompt, completion) pairs from random.Random(seed)."""
    rng = random.Random(seed)
    demos = []
    for _ in range(n):
        prompt = "".join(LETTERS[rng.randrange(len(LETTERS))] for _ in range(length))
        completion = write_answer(prompt, rng, p_copy, p_typo)
        demos.append((prompt, completion))
    return demos
