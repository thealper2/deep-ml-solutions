def reward_honest_participants(balances, worker_id, votes, verdict, reward_worker, reward_verifier):
    new_balances = balances.copy()

    if verdict:
        new_balances[worker_id] = new_balances.get(worker_id, 0.0) + reward_worker

    for vote in votes:
        verifier_id = vote['verifier_id']
        if vote['vote'] == verdict:
            new_balances[verifier_id] = new_balances.get(verifier_id, 0.0) + reward_verifier

    return new_balances
