import random


def make_task(rng, lo=1, hi=9, max_target=999):
    while True:
        nums = (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))
        cand = rng.choice(candidates(nums))
        val = eval_rpn(cand)
        if val is not None and 0 <= val <= max_target:
            return (nums, val)