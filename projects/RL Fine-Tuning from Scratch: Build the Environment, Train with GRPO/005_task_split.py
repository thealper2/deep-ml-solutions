import random


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    seen = set()
    distinct = []
    for _ in range(n_draws):
        task = make_task(rng, lo, hi, max_target)
        if task not in seen:
            seen.add(task)
            distinct.append(task)

    rng.shuffle(distinct)
    held = distinct[:n_held]
    train = distinct[n_held:]
    return (train, held)