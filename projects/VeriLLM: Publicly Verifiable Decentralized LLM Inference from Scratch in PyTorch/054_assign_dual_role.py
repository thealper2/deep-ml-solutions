def assign_dual_role(node_ids, worker_id, committee_size, seed):
    if worker_id not in node_ids:
        node_ids = node_ids + [worker_id]

    other_ids = [n for n in node_ids if n != worker_id]

    if committee_size == 1:
        return {'worker_id': worker_id, 'committee': [worker_id]}

    if len(other_ids) < committee_size - 1:
        committee = other_ids.copy()
        while len(committee) < committee_size:
            committee.append(worker_id)

        return {'worker_id': worker_id, 'committee': committee}

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(other_ids), size=committee_size - 1, replace=False)
    committee = [other_ids[i] for i in indices]
    committee.append(worker_id)

    rng.shuffle(committee)

    return {'worker_id': worker_id, 'committee': committee}
