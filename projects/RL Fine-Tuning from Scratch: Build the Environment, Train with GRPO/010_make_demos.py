import random


def make_demos(rng, tasks):
    demos = []
    for nums, target in tasks:
        sols = solve(nums, target)
        chosen = rng.choice(sols)
        prompt = encode_task(nums, target)
        completion = [TOK[t] for t in chosen] + [EOS]
        demos.append((prompt, completion))

    return demos