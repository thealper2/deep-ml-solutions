def aggregate_votes_majority(votes):
    accept_count = sum(1 for v in votes if v['vote'])
    reject_count = len(votes) - accept_count
    verdict = accept_count > reject_count

    return {
        'verdict': verdict,
        'accept_count': accept_count,
        'reject_count': reject_count,
    }
