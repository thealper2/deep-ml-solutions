import numpy as np


def accuracy_and_soundness(outcomes):
    n = len(outcomes)
    correct = sum(1 for o in outcomes if o == 'correct')
    sound = sum(1 for o in outcomes if o in ('correct', 'abstain'))
    return (round(correct / n, 4), round(sound / n, 4))


def sequential_cost(winner, winner_round, own_rounds):
    return sum(own_rounds[:winner]) + winner_round


def forward_percentiles(costs, ps):
    return {p: float(np.percentile(costs, p)) for p in ps}


def solve_rate_within_budget(costs, solved, budgets):
    result = []
    for b in budgets:
        count = sum(1 for c, s in zip(costs, solved) if s and c <= b)
        result.append(count / len(costs))

    return result


def chain_success_probability(p, K):
    return 1 - (1 - p) ** K


def expected_attempts(alive_counts):
    product = 1.0
    for c in alive_counts:
        product *= c

    return float(product)
