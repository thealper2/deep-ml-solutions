def slash_worker(balances, worker_id, slash_amount):
    new_balances = balances.copy()
    current = new_balances.get(worker_id, 0.0)
    new_balances[worker_id] = current - slash_amount
    return new_balances
