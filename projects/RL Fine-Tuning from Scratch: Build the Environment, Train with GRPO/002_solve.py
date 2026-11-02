import itertools

OPS = ['+', '-', '*']

def candidates(nums):
    result = []
    for perm in itertools.permutations(nums):
        a, b, c = (str(x) for x in perm)
        for o1 in OPS:
            for o2 in OPS:
                result.append([a, b, o1, c, o2])
                result.append([a, b, c, o1, o2])

    return result


def solve(nums, target):
    seen = set()
    result = []
    for cand in candidates(nums):
        key = tuple(cand)
        if key in seen:
            continue

        seen.add(key)
        if eval_rpn(cand) == target:
            result.append(cand)

    return result