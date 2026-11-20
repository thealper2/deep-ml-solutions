def run_trials(policy, tasks, store, n_trials, max_calls=4):
    """outcomes[i][j] = 1 if task i was solved in trial j; trials run one after another."""
    outcomes = [[0] * n_trials for _ in range(len(tasks))]
    for j in range(n_trials):
        for i, task in enumerate(tasks):
            ep = run_episode(policy, task, store, max_calls)
            outcomes[i][j] = 1 if ep["correct"] else 0

    return outcomes